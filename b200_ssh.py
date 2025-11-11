import os, sys, time, json, subprocess, shlex, signal
import requests

RUNPOD_API = "https://rest.runpod.io/v1"
API_KEY = os.getenv("RUNPOD_API_KEY") or "<PUT_API_KEY_HERE>"
SSH_KEY_PRIV = os.getenv("RUNPOD_SSH_KEY_PATH", os.path.expanduser("~/.ssh/runpodprivate"))
SSH_KEY_PUB  = os.getenv("RUNPOD_SSH_PUB_PATH", SSH_KEY_PRIV + ".pub")
LOCAL_FILE_TO_SYNC = sys.argv[1] if len(sys.argv) > 1 else None  # optional: python runpod_b200_ncu.py ./my_kernel.cu

# Pod config (spot B200, no persistent volume, public IP for SSH)
IMAGE = os.getenv("RUNPOD_IMAGE", "nvidia/cuda:12.9.0-devel-ubuntu22.04")
GPU_TYPE = "NVIDIA B200"  # see RunPod GPU types doc for exact ID
CREATE_BODY = {
    "cloudType": "COMMUNITY",
    "computeType": "GPU",
    "interruptible": True,                  # spot
    "gpuCount": 1,
    "gpuTypeIds": [GPU_TYPE],
    "allowedCudaVersions": ["12.9"],        # CUDA 13 for B200 / Nsight Compute 2025.3
    "containerDiskInGb": 30,                # ephemeral container disk
    "volumeInGb": 0,                        # no persistent volume
    "imageName": IMAGE,
    "name": "b200-ncu-spot",
    "supportPublicIp": True,                # needed for raw TCP/SSH on Community Cloud
    "ports": ["22/tcp"],
    # Start sshd and make sure ncu is there; sleep infinity keeps the container alive
    "dockerStartCmd": [
        "bash","-lc",
        r"""
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y openssh-server curl gnupg
mkdir -p /root/.ssh; chmod 700 /root/.ssh
echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
service ssh start

# Ensure Nsight Compute CLI 'ncu' exists (CUDA 13 devel images usually include it)
if ! command -v ncu >/dev/null 2>&1; then
  # Add CUDA apt repo keyring if missing, then install cuda-nsight-compute-13-0
  if ! ls /etc/apt/sources.list.d | grep -qi cuda; then
    . /etc/os-release
    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/$(dpkg --print-architecture)/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update
  fi
  apt-get install -y cuda-nsight-compute-13-0 || true
fi
ncu --version || true
sleep infinity
        """.strip()
    ],
    "env": {}  # we fill PUBLIC_KEY below
}

HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def die(msg, code=1):
    print(msg, file=sys.stderr); sys.exit(code)

def read_pubkey(path):
    with open(path, "r") as f: return f.read().strip()

def create_pod(pubkey):
    body = dict(CREATE_BODY)
    env = dict(body.get("env", {}))
    env["PUBLIC_KEY"] = pubkey
    body["env"] = env
    r = requests.post(f"{RUNPOD_API}/pods", headers=HEADERS, data=json.dumps(body), timeout=90)
    if r.status_code >= 300:
        die(f"Create pod failed: {r.status_code} {r.text}")
    return r.json()

def get_pod(pod_id):
    r = requests.get(f"{RUNPOD_API}/pods/{pod_id}", headers=HEADERS, timeout=20)
    if r.status_code >= 300:
        die(f"Get pod failed: {r.status_code} {r.text}")
    return r.json()

def delete_pod(pod_id):
    try:
        requests.delete(f"{RUNPOD_API}/pods/{pod_id}", headers=HEADERS, timeout=30)
    except Exception:
        pass

def wait_for_ssh(pod_id, timeout=900):
    # Wait until publicIp and portMappings["22"] are present
    t0 = time.time()
    while True:
        pod = get_pod(pod_id)
        ip = pod.get("publicIp")
        pm = pod.get("portMappings") or {}
        ssh_port = pm.get("22")
        if ip and ssh_port:
            return ip, int(ssh_port)
        if time.time() - t0 > timeout:
            die("Timed out waiting for SSH mapping on pod.")
        time.sleep(5)

def run(cmd):
    print("+", cmd)
    return subprocess.call(cmd, shell=True)

def scp_file(ip, port, local_path, key_path):
    local_path = os.path.abspath(local_path)
    cmd = f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {shlex.quote(key_path)} -P {port} {shlex.quote(local_path)} root@{ip}:/root/"
    return run(cmd)

def ssh_shell(ip, port, key_path):
    cmd = f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {shlex.quote(key_path)} -p {port} root@{ip}"
    return run(cmd)

def main():
    if not API_KEY or API_KEY.startswith("<PUT_API_KEY_HERE>"):
        die("Set RUNPOD_API_KEY or hardcode API_KEY in this script.")
    if not os.path.exists(SSH_KEY_PRIV): die(f"Missing SSH private key at {SSH_KEY_PRIV}")
    if not os.path.exists(SSH_KEY_PUB):  die(f"Missing SSH public key at {SSH_KEY_PUB}")
    pubkey = read_pubkey(SSH_KEY_PUB)

    print("Creating spot B200 pod...")
    pod = create_pod(pubkey)
    pod_id = pod["id"]
    print("Pod ID:", pod_id)

    # Ensure we clean up on Ctrl-C
    def _cleanup(sig=None, frame=None):
        print("\nTearing down pod...")
        delete_pod(pod_id)
        sys.exit(0)
    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    print("Waiting for public IP + SSH port...")
    ip, port = wait_for_ssh(pod_id)
    print(f"SSH: ssh -i {SSH_KEY_PRIV} -p {port} root@{ip}")

    if LOCAL_FILE_TO_SYNC:
        print(f"Syncing {LOCAL_FILE_TO_SYNC} -> /root/ on pod")
        scp_file(ip, port, LOCAL_FILE_TO_SYNC, SSH_KEY_PRIV)

    print("Dropping into root shell. Exit shell to auto-destroy the pod.")
    ssh_shell(ip, port, SSH_KEY_PRIV)

    print("Destroying pod...")
    delete_pod(pod_id)
    print("Done.")

if __name__ == "__main__":
    main()