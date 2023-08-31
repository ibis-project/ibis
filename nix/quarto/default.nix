{ stdenv
, lib
, esbuild
, deno
, fetchurl
, dart-sass
, makeWrapper
, python3
, extraPythonPackages ? ps: with ps; [ ]
}:

stdenv.mkDerivation rec {
  pname = "quarto";
  version = "1.4.339";
  src = fetchurl {
    url = "https://github.com/quarto-dev/quarto-cli/releases/download/v${version}/quarto-${version}-linux-amd64.tar.gz";
    sha256 = "sha256-Toh+sImbIGc7PK4UUq9/p8nVBANkcOA6yR3fF6Nk76M=";
  };

  nativeBuildInputs = [ makeWrapper ];

  preFixup = ''
    wrapProgram $out/bin/quarto \
      --prefix QUARTO_ESBUILD : ${esbuild}/bin/esbuild \
      --prefix QUARTO_DENO : ${deno}/bin/deno \
      --prefix QUARTO_DART_SASS : ${dart-sass}/bin/dart-sass \
      --prefix QUARTO_PYTHON : ${python3.withPackages (ps: [ ps.jupyter ] ++ (extraPythonPackages ps))}/bin/python3
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
    maintainers = with maintainers; [ mrtarantoga ];
    platforms = [ "x86_64-linux" ];
    sourceProvenance = with sourceTypes; [ binaryNativeCode binaryBytecode ];
  };
}
