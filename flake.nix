{
  description = "My Python Project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix-src = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix-src }: 
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        overlays = [ poetry2nix-src.overlay ];
        python = pkgs.python311;
        project = pkgs.poetry2nix.mkPoetryApplication {
          projectDir = ./.;
        };
      in {
        packages.default = project;
        devShells.default = pkgs.mkShell {
          buildInputs = [ project ];
          nativeBuildInputs = [ pkgs.poetry ];
        };
      }
    );
}
