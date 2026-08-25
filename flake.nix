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
          gdb
          valgrind
          kdePackages.kcachegrind
        ];

        shellHook = ''
          unset NIX_ENFORCE_NO_NATIVE
        '';
      };
    };
}
