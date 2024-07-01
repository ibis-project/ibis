final: prev: {
  scipy = prev.scipy.overridePythonAttrs (
    old: final.lib.optionalAttrs (final.stdenv.isDarwin && (final.lib.versionAtLeast old.version "1.14.0")) {
      prePatch = (old.prePatch or "") + ''
        substituteInPlace scipy/meson.build \
          --replace xcrun "${final.pkgs.xcbuild}/bin/xcrun"
      '';
    }
  );
}
