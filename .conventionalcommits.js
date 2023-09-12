"use strict";

const releaseConfig = require("./.releaserc.js");

function extractConventionalCommitsConfig(config) {
  return config.plugins
    .filter(
      ([plugin, _]) => plugin == "@semantic-release/release-notes-generator",
    )
    .map(([_, config]) => config)[0].presetConfig.types;
}

module.exports = {
  options: {
    preset: {
      name: "conventionalcommits",
      types: extractConventionalCommitsConfig(releaseConfig),
    },
  },
};
