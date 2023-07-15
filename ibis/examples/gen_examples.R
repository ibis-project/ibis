library(tidyverse)
library(stringr)
library(palmerpenguins)
library(janitor)

LOOKUP <- list("penguins_raw (penguins)" = "penguins_raw")
IGNORED <- c("sim1", "sim2", "sim3", "sim4", "table1", "table2", "table3", "table4a", "table4b", "table5", "Animals", "Oats", "Muscle", "Melanoma")
DESCRIPTIONS_PATH <- "ibis/examples/descriptions"
DATA_PATH <- "ibis/examples/data"

RESULTS <- as.data.frame(data(package = .packages(all.available = TRUE))$results)

write_description <- function (name, description) {
    cat(description, file = paste(DESCRIPTIONS_PATH, name, sep = "/"))
}

for (i in 1:nrow(RESULTS)) {
    row <- RESULTS[i,]
    package <- row$Package

    library(package, warn.conflicts = FALSE, character.only = TRUE)

    item <- row$Item
    name <- LOOKUP[[item]]

    if (is.null(name)) {
        name <- item
    }

    data <- tryCatch(get(name), error = function (cond) return(NULL))

    name <- str_replace_all(name, "\\.", "_")

    if (!(name %in% IGNORED) && !is.null(data) && is.data.frame(data)) {
        basename <- paste(name, "csv.gz", sep = ".")
        file <- paste(DATA_PATH, basename, sep = "/")

        clean_data <- clean_names(data)
        write_csv(clean_data, file = file, quote = "needed", na = "")

        description <- row$Title
        write_description(name, description)

        # write a raw-column-name file if the clean names differ
        if (any(names(clean_data) != names(data))) {
            raw_name <- paste(name, "raw", sep = "_")
            raw_file <- paste(DATA_PATH, paste(raw_name, "csv.gz", sep = "."), sep = "/")

            write_csv(data, file = raw_file, quote = "needed", na = "")
            write_description(raw_name, description)
        }
    }
}
