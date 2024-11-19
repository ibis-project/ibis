{ stdenv
, lib
, esbuild
, fetchurl
, dart-sass
, makeWrapper
, rWrapper
, rPackages
, autoPatchelfHook
}:

let
  platforms = {
    "x86_64-linux" = "linux-amd64";
    "aarch64-linux" = "linux-arm64";
    "aarch64-darwin" = "macos";
  };
  shas = {
    "x86_64-linux" = "sha256-mVoFBQJJHGn5ZbwOtamshEQl9FzmRVEBye3bBXFUlUI=";
    "aarch64-linux" = "sha256-TNik4+OdDqGwArw9wkrq4wNHt6tGgYo32V9KNPSsPWo=";
    "aarch64-darwin" = "sha256-fjcmyVyPSHyHBICjpweuCnGtMAAlPNNzBMHEk+2emBA=";
  };
  inherit (stdenv.hostPlatform) system;
in
stdenv.mkDerivation rec {
  pname = "quarto";
  version = "1.6.32";
  src = fetchurl {
    url = "https://github.com/quarto-dev/quarto-cli/releases/download/v${version}/quarto-${version}-${platforms.${system}}.tar.gz";
    sha256 = shas.${system};
  };

  preUnpack = lib.optionalString stdenv.isDarwin "mkdir ${sourceRoot}";
  sourceRoot = lib.optionalString stdenv.isDarwin "quarto-${version}";
  unpackCmd = lib.optionalString stdenv.isDarwin "tar xzf $curSrc --directory=$sourceRoot";

  nativeBuildInputs = lib.optionals stdenv.isLinux [ autoPatchelfHook ] ++ [ makeWrapper ];

  preFixup =
    let
      rEnv = rWrapper.override {
        packages = with rPackages; [ dplyr reticulate rmarkdown tidyr ];
      };
    in
    ''
      wrapProgram $out/bin/quarto \
        --prefix QUARTO_ESBUILD : ${lib.getExe esbuild} \
        --prefix QUARTO_R : ${lib.getExe' rEnv "R"} \
        --prefix QUARTO_DART_SASS : ${lib.getExe dart-sass}
    '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin $out/share

    mv bin/* $out/bin
    mv share/* $out/share

    runHook postInstall
  '';

  meta = with lib; {
    description = "Open-source scientific and technical publishing system built on Pandoc";
    longDescription = ''
      Quarto is an open-source scientific and technical publishing system built on Pandoc.
      Quarto documents are authored using markdown, an easy to write plain text format.
    '';
    homepage = "https://quarto.org/";
    changelog = "https://github.com/quarto-dev/quarto-cli/releases/tag/v${version}";
    license = licenses.gpl2Plus;
    platforms = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
    sourceProvenance = with sourceTypes; [ binaryNativeCode binaryBytecode ];
  };
}
