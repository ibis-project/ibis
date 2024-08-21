{ stdenv
, lib
, esbuild
, deno
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
    "x86_64-linux" = "sha256-1HqjMENJ1H5RBaKIRZoUDGrdSEQOhhIiRLIQFqnkFlk=";
    "aarch64-linux" = "sha256-kRh8wC4nzHuE25Ymj+e+cy6+cl0YYF44cUAZwbjCCvw=";
    "aarch64-darwin" = "sha256-/1gV3MrRNuGPALzR+1o3x9BvZakyZDRYQwRUzagfHMA=";
  };
  inherit (stdenv.hostPlatform) system;
in
stdenv.mkDerivation rec {
  pname = "quarto";
  version = "1.5.55";
  src = fetchurl {
    url = "https://github.com/quarto-dev/quarto-cli/releases/download/v${version}/quarto-${version}-${platforms.${system}}.tar.gz";
    sha256 = shas.${system};
  };

  preUnpack = lib.optionalString stdenv.isDarwin "mkdir ${sourceRoot}";
  sourceRoot = lib.optionalString stdenv.isDarwin "quarto-${version}";
  unpackCmd = lib.optionalString stdenv.isDarwin "tar xzf $curSrc --directory=$sourceRoot";

  nativeBuildInputs = lib.optionals stdenv.isLinux [ autoPatchelfHook ] ++ [ makeWrapper ];

  preFixup = ''
    wrapProgram $out/bin/quarto \
      --prefix QUARTO_ESBUILD : ${esbuild}/bin/esbuild \
      --prefix QUARTO_DENO : ${deno}/bin/deno \
      --prefix QUARTO_R : ${rWrapper.override { packages = with rPackages; [ dplyr reticulate rmarkdown tidyr ]; }}/bin/R \
      --prefix QUARTO_DART_SASS : ${dart-sass}/bin/dart-sass
  '';

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin $out/share

    rm -r bin/tools/*/deno*

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
    maintainers = with maintainers; [ mrtarantoga ];
    platforms = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
    sourceProvenance = with sourceTypes; [ binaryNativeCode binaryBytecode ];
  };
}
