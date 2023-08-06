import subprocess


def install_apt_package(package_name):
    try:
        # Use subprocess.run() to execute the apt-get command with root privileges (sudo)
        subprocess.run(['sudo', 'apt-get', 'update'], check=True,
                       capture_output=True)  # Update package lists

        # Install the specified package
        subprocess.run(['sudo', 'apt-get', 'install', '-y',
                       package_name], check=True, capture_output=True)

        print(f"{package_name} installed successfully.")
    except subprocess.CalledProcessError as e:
        # Handle any errors that might occur during the installation process
        print(f"Error while installing {package_name}:")
        print(e.stderr.decode())


if __name__ == "__main__":
    # Replace with the name of the package you want to install
    package_to_install = "libgl1-mesa-glx"
    install_apt_package(package_to_install)
