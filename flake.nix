{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          cudaPackages.cuda_nvcc
          cudaPackages.cuda_cudart
          cudaPackages.libcublas

          gdb
          valgrind
          kdePackages.kcachegrind

          cudaPackages.nsight_systems
          cudaPackages.nsight_compute
          cudaPackages.cuda_gdb
          cudaPackages.cuda_sanitizer_api
          cudaPackages.cuda_nvtx
          cudaPackages.cuda_cuobjdump
          cudaPackages.cuda_nvdisasm
        ];

        shellHook = ''
          unset NIX_ENFORCE_NO_NATIVE
        '';
      };
    };
}
