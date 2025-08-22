{ stdenv, lib, esbuild, fetchurl, dart-sass, makeWrapper, rWrapper, rPackages
, autoPatchelfHook, libgcc, which, }:

let
  platforms = rec {
    x86_64-linux = "linux-amd64";
    aarch64-linux = "linux-arm64";
    aarch64-darwin = "macos";
  };

  inherit (stdenv.hostPlatform) system;
  versionInfo = builtins.fromJSON (builtins.readFile ./version-info.json);
in stdenv.mkDerivation rec {
  pname = "quarto";
  inherit (versionInfo) version;
  src = fetchurl {
    url =
      "https://github.com/quarto-dev/quarto-cli/releases/download/v${version}/quarto-${version}-${
        platforms.${system}
      }.tar.gz";
    sha256 = versionInfo.hashes.${system};
  };

  preUnpack = lib.optionalString stdenv.isDarwin "mkdir ${sourceRoot}";
  sourceRoot = lib.optionalString stdenv.isDarwin "quarto-${version}";
  unpackCmd = lib.optionalString stdenv.isDarwin
    "tar xzf $curSrc --directory=$sourceRoot";

  nativeBuildInputs = lib.optionals stdenv.isLinux [ autoPatchelfHook ]
    ++ [ makeWrapper libgcc ];

  preFixup = let
    rEnv = rWrapper.override {
      packages = with rPackages; [ dplyr reticulate rmarkdown tidyr ];
    };
  in ''
    wrapProgram $out/bin/quarto \
      --prefix QUARTO_ESBUILD : ${lib.getExe esbuild} \
      --prefix QUARTO_R : ${lib.getExe' rEnv "R"} \
      --prefix QUARTO_DART_SASS : ${lib.getExe dart-sass} \
      --prefix PATH : ${lib.makeBinPath [ which ]}
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin $out/share

    mv bin/* $out/bin
    mv share/* $out/share

    runHook postInstall
  '';

  meta = with lib; {
    description =
      "Open-source scientific and technical publishing system built on Pandoc";
    longDescription = ''
      Quarto is an open-source scientific and technical publishing system built on Pandoc.
      Quarto documents are authored using markdown, an easy to write plain text format.
    '';
    homepage = "https://quarto.org/";
    changelog =
      "https://github.com/quarto-dev/quarto-cli/releases/tag/v${version}";
    license = licenses.gpl2Plus;
    platforms = builtins.attrNames platforms;
    sourceProvenance = with sourceTypes; [ binaryNativeCode binaryBytecode ];
  };
}
