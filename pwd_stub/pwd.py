"""Minimal Windows stub for the Unix pwd module."""
import os

class struct_passwd:
    def __init__(self, name, uid, gid, dir, shell):
        self.pw_name = name; self.pw_uid = uid; self.pw_gid = gid
        self.pw_dir = dir; self.pw_shell = shell

def getpwuid(uid):
    return struct_passwd(os.environ.get("USERNAME", "user"), uid, 0,
                        os.path.expanduser("~"), "")

def getpwnam(name):
    return struct_passwd(name, os.getuid() if hasattr(os, "getuid") else 0,
                        0, os.path.expanduser("~"), "")
