{ fetchFromGitHub
, lib
, stdenv
, arrow-cpp
, cmake
, gtest
, ninja
, sqlite
, pname
, sourceRoot
, installCheckPhase ? ""
, doInstallCheck ? false
}:
stdenv.mkDerivation rec {
  inherit pname;
  version = "0.0.1"; # not actually a version (yet)
  src = fetchFromGitHub {
    owner = "apache";
    repo = "arrow-adbc";
    rev = "ef70af6aff8da61bbe2f19f14c607aa6c07caf33";
    hash = "sha256-ibk30T0EXbI5ZB5so0+TV6SCiIC7M4taAqUL0ohW4us=";
  };

  inherit sourceRoot;

  nativeBuildInputs = [ cmake ninja ];
  buildInputs = [ arrow-cpp sqlite ] ++ lib.optionals doInstallCheck [ gtest ];

  cmakeFlags = [
    "-DADBC_BUILD_TESTS=${if doInstallCheck then "ON" else "OFF"}"
    "-DADBC_BUILD_SHARED=ON"
    "-DADBC_BUILD_STATIC=OFF"
  ];

  inherit doInstallCheck installCheckPhase;

  meta = with lib; {
    description = "Arrow Database Connectivity";
    homepage = "https://github.com/apache/arrow-adbc";
    license = licenses.asl20;
    platforms = platforms.unix;
    maintainers = with maintainers; [ cpcloud ];
  };
}
