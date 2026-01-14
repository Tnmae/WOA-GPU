{
  description = "CUDA development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
        cudaVersion = "12";
      };
    };
  in {
    formatter.${system} = pkgs.alejandra;

    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        cudatoolkit
        cudaPackages.cudnn

        # Tooling
        gcc
        gdb
        cmake
        binutils
        pkg-config

        # OpenGL / X11 (for CUDA-GL interop)
        libGL
        libGLU
        freeglut
        xorg.libX11
        xorg.libXext
        xorg.libXi
        xorg.libXmu
        xorg.libXrandr
       xorg.libXv

        # Misc
        zlib
        ncurses
        fmt.dev
        ffmpeg
        uv
      ];

      shellHook = ''
        export CUDA_HOME=${pkgs.cudatoolkit}
        export CUDA_PATH=${pkgs.cudatoolkit}

        # Required on NixOS for libcuda.so
        export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH

        export CMAKE_PREFIX_PATH="${pkgs.fmt.dev}:$CMAKE_PREFIX_PATH"
        export PKG_CONFIG_PATH="${pkgs.fmt.dev}/lib/pkgconfig:$PKG_CONFIG_PATH"

        echo "CUDA dev shell ready"
        nvcc --version
      '';
    };
  };
}

