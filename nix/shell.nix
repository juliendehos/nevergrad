# nix-shell -- run "python -m nevergrad.benchmark illcond --seed=12 --repetitions=1 --plot"

with import <nixpkgs> { config = { allowBroken = true; }; };

(let

  python = let

    packageOverrides = self: super: {

      bayesian-optimization = self.buildPythonPackage rec {
        pname = "bayesian-optimization";
        version = "1.2.0";
        src = self.fetchPypi {
          inherit pname version;
          sha256 = "0df95dqq8mwdb8cfh96rr9av9fgz02lv55aj2hffw96cnvs3mzf2";
        };
        propagatedBuildInputs = [
          self.scikitlearn
          self.scipy
        ];
      };

      lpips = self.buildPythonPackage rec {
        pname = "lpips";
        version = "0.1.3";
        src = fetchFromGitHub { 
          owner = "richzhang";
          repo = "PerceptualSimilarity";
          rev = "v${version}";
          sha256 = "0dakgc5rahblqclabhy04f65ms5g2hhs2cq034mpya2dl0bcsfmk";
        }; 
        doCheck = false;
      };

      mixsimulator = self.buildPythonPackage rec {
        pname = "mixsimulator";
        version = "0.2.9.9";
        src = self.fetchPypi {
          inherit pname version;
          sha256 = "1nz3lzvdl4nzv3c6cysz7m5clxmhihmhmdwqzsll8j3sp2h2w60k";
        };
        doCheck = false;
      };

      libsvm = self.buildPythonPackage rec {
        pname = "libsvm";
        version = "3.23.0.4";
        src = self.fetchPypi {
          inherit pname version;
          sha256 = "0qq444h9vikz63fjhs0fhcr6zxm02j32glyqx1ymv9xsnx4cfqga";
        };
      };

      image-quality = self.buildPythonPackage rec {
        pname = "image-quality";
        version = "1.2.6";
        src = self.fetchPypi {
          inherit pname version;
          sha256 = "1cp4nh08isq71njl2psw94210gm19vbjahmil0kyi1kljirwvrpz";
        };
        doCheck = false;
        propagatedBuildInputs = [
          self.scipy
          self.scikitimage
          self.libsvm
        ];
      };

      nevergrad = self.buildPythonPackage {
        pname = "nevergrad";
        version = "git";
        src = ./.;
        #buildInputs = [self.pytest];
        doCheck = false;
        propagatedBuildInputs = [
          self.bayesian-optimization
          self.cma
          self.gym 
          self.matplotlib
          self.image-quality
          self.lpips
          self.mixsimulator
          self.numpy
          self.opencv4
          self.pandas
          self.pyproj
          self.pytorch
          self.torchvision
          self.tqdm
          self.typing-extensions

          self.pygments
          self.recommonmark
          self.sphinx
          self.sphinx_rtd_theme
        ];
      };

    };

  in pkgs.python3.override { inherit packageOverrides; self = python; };

in python.withPackages(ps: [ ps.nevergrad ])).env

