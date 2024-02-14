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

stdenv.mkDerivation rec {
  pname = "quarto";
  version = "1.5.13";
  src = fetchurl {
    url = "https://github.com/quarto-dev/quarto-cli/releases/download/v${version}/quarto-${version}-linux-amd64.tar.gz";
    sha256 = "sha256-X+VgTY649Vo37u8byNzLD+KPVK3MRdySAPN0ZhdBw0g=";
  };

  nativeBuildInputs = [ autoPatchelfHook makeWrapper ];

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
    platforms = [ "x86_64-linux" ];
    sourceProvenance = with sourceTypes; [ binaryNativeCode binaryBytecode ];
  };
}
