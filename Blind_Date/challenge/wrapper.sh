#!/bin/bash

echo [+] Starting challenge on port 1337
socat TCP-LISTEN:1337,reuseaddr,fork EXEC:"./chall"
