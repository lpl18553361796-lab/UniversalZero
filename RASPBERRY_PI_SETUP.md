# Raspberry Pi Setup

This project is intended to run on Raspberry Pi as a demo and inference app.
Full training is much better on a laptop or desktop machine.

## Recommended device

- Raspberry Pi 5, 8 GB RAM recommended
- 64-bit Raspberry Pi OS
- 32 GB or larger SD card, or SSD

## First-time setup

Copy the whole project folder to the Raspberry Pi, then open a terminal in the project root:

```bash
cd /home/lpl/UniversalZero-main/UniversalZero-main
```

Install basic Python tools if needed:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-torch
```

Run the installer:

```bash
chmod +x /home/lpl/UniversalZero-main/UniversalZero-main/raspberry_pi/lazy_install_all.sh
bash /home/lpl/UniversalZero-main/UniversalZero-main/raspberry_pi/lazy_install_all.sh
```

The installer will:

- create `.venv-pi`
- use apt-installed CPU PyTorch through system site packages
- configure pip to use the Tsinghua PyPI mirror
- install `requirements-pi.txt`
- create a desktop launcher named `UniversalZero`
- retry pip downloads when the network is interrupted
- wait if apt or dpkg is temporarily locked by another system update

## Daily use

Double-click `UniversalZero` on the Raspberry Pi desktop.

It starts the Streamlit UI at:

```text
http://localhost:8501
```

## Notes

Python is still required. Raspberry Pi does not use Windows `.exe` files, and bundling PyTorch into a single executable is usually larger and less reliable than a virtual environment.

For Raspberry Pi, use low MCTS settings for demos. Avoid full training runs on the Pi.
