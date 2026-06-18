{
  description = "Project 1 (BC -> ALICE -> ACT) dev env for GTX 1060 (Pascal sm_61) on Ubuntu";

  # Pinned nixpkgs. nixos-25.05 ships python310 (robomimic/robosuite want 3.10)
  # AND its glibc 2.40 carries the backported vDSO patch needed for kernel 6.17
  # (nixos-24.11's glibc 2.40 lacks it -> "__vdso_gettimeofday: invalid mode for dlopen()").
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # Native libraries MuJoCo / robosuite / rendering need at runtime.
      # These replace the `sudo apt install ...` step from the setup doc.
      nativeLibs = with pkgs; [
        stdenv.cc.cc.lib   # libstdc++ / libgomp for manylinux wheels (numba, opencv)
        zlib
        glib               # libglib-2.0 / libgthread-2.0 -- needed by opencv-python (cv2)
        libGL              # libglvnd's libGL
        libGLU
        glew               # libGLEW (robosuite onscreen)
        glfw               # libglfw3 (default GLFW backend / onscreen viewer)
        mesa               # GLX / software fallback
        libglvnd           # GL vendor dispatch
        xorg.libX11
        xorg.libXrandr
        xorg.libXinerama
        xorg.libXcursor
        xorg.libXi
        xorg.libXext
        xorg.libXfixes
        xorg.libXrender    # opencv-python / X rendering
        xorg.libSM
        xorg.libICE
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        # Build/dev tools available on PATH inside the shell.
        nativeBuildInputs = with pkgs; [
          python310
          uv                 # fast venv + pip resolver for the ML wheels
          git
          wget
          curl
          patchelf
          cmake              # egl-probe (robomimic dep) builds a C++ ext with cmake
          ffmpeg
          glxinfo            # mesa-utils equivalent, to test GL
          xorg.xeyes         # quick GUI/display test (x11-apps equivalent)
        ];

        # nativeLibs for runtime; linuxHeaders so C extensions that compile against
        # the kernel uAPI (e.g. evdev, pulled in by robosuite->pynput) see a recent,
        # consistent <linux/*.h> instead of glibc's older bundled copy ("KEY_LINK_PHONE
        # undeclared"). pkg-config helps source builds find libs.
        buildInputs = nativeLibs ++ [ pkgs.linuxHeaders pkgs.pkg-config ];

        shellHook = ''
          # --- Runtime library search path -------------------------------------
          # Curated NVIDIA driver libs ONLY. We must NOT put all of
          # /usr/lib/x86_64-linux-gnu on LD_LIBRARY_PATH: it contains the SYSTEM
          # glibc, which would shadow the Nix glibc for every Nix binary and make
          # them die with "__vdso_gettimeofday: invalid mode for dlopen()".
          # Instead, symlink just the driver .so's (libcuda for torch; the nvidia
          # GL/EGL vendor libs for later hardware rendering) into a local dir.
          DRIVER_LIBS="$PWD/.driver-libs"
          mkdir -p "$DRIVER_LIBS"
          for _l in \
            /usr/lib/x86_64-linux-gnu/libcuda.so* \
            /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
            /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* \
            /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so* ; do
            [ -e "$_l" ] && ln -sf "$_l" "$DRIVER_LIBS/"
          done
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath nativeLibs}:$DRIVER_LIBS"

          # --- Kernel uAPI headers for C-extension builds (evdev) ----------------
          # evdev's generator scans /usr/include/linux (kernel 6.17 -> has new KEY_*
          # macros) while the Nix gcc otherwise compiles against older bundled
          # headers -> "KEY_LINK_PHONE undeclared". Expose ONLY the system linux/
          # uAPI to the compiler at high priority so generation and compilation agree
          # (glibc's own headers stay Nix-provided, since we don't expose all of
          # /usr/include).
          SYS_HEADERS="$PWD/.sys-headers"
          mkdir -p "$SYS_HEADERS"
          ln -sfn /usr/include/linux "$SYS_HEADERS/linux"
          export NIX_CFLAGS_COMPILE="-isystem $SYS_HEADERS ''${NIX_CFLAGS_COMPILE:-}"

          # MuJoCo rendering backend:
          #   - LEAVE MUJOCO_GL UNSET for the on-screen GLFW viewer (live window).
          #   - For headless offscreen video, prefix the command:  MUJOCO_GL=egl python make_video.py
          # State-based BC training needs NO rendering at all, so unset is correct.
          unset MUJOCO_GL

          export UV_PYTHON="${pkgs.python310}/bin/python3.10"

          # --- Project-local venv ----------------------------------------------
          # The Nix python lives in /nix/store (read-only) -> we can't pip-install
          # into it. A venv is the writable layer for the pip wheels. On first
          # entry we auto-build it; a sentinel keeps it from re-running.
          export VENV_DIR="$PWD/.venv"
          if [ -z "''${SKIP_BOOTSTRAP:-}" ] && [ ! -f "$VENV_DIR/.bootstrap_ok" ]; then
            echo "[flake] First run -> bootstrapping Python stack (downloads torch cu126, etc.)..."
            if [ -x "$PWD/bootstrap_venv.sh" ]; then
              "$PWD/bootstrap_venv.sh" || echo "[flake] bootstrap failed; re-enter 'nix develop' to retry."
            else
              echo "[flake] bootstrap_venv.sh missing/not executable; skipping."
            fi
          fi
          if [ -f "$VENV_DIR/bin/activate" ]; then
            # shellcheck disable=SC1091
            source "$VENV_DIR/bin/activate"
            echo "[flake] activated $VENV_DIR"
          fi

          echo "[flake] devShell ready. Python: $(python3.10 --version 2>&1). GPU below:"
          nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not found)"
        '';
      };
    };
}
