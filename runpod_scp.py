#!/usr/bin/env python3
import os, sys, subprocess, shlex

# Default values
DEFAULT_IP = "38.80.152.76"
DEFAULT_PORT = "30406"
SSH_KEY_PATH = os.path.expanduser("~/.ssh/runpodprivate")
LOCAL_FILE = "sub_pod.py"

def run(cmd):
    """Run a shell command and print it"""
    print("+", cmd)
    return subprocess.call(cmd, shell=True)

def scp_file(ip, port, local_path, key_path, remote_path):
    """SCP a file to remote server"""
    local_path = os.path.abspath(local_path)
    cmd = f"scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {shlex.quote(key_path)} -P {port} {shlex.quote(local_path)} root@{ip}:{remote_path}"
    return run(cmd)

def main():
    # Parse command line arguments
    ip = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IP
    port = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PORT

    # Check if SSH key exists
    if not os.path.exists(SSH_KEY_PATH):
        print(f"Error: SSH private key not found at {SSH_KEY_PATH}")
        sys.exit(1)

    # Check if submission.py exists
    if not os.path.exists(LOCAL_FILE):
        print(f"Error: {LOCAL_FILE} not found in current directory")
        sys.exit(1)

    print(f"SCP {LOCAL_FILE} to root@{ip}:{port}:/root/")
    result = scp_file(ip, port, LOCAL_FILE, SSH_KEY_PATH, "/root/")

    if result == 0:
        print("File transfer successful!")
    else:
        print("File transfer failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
