library(tidyverse)
library(stringr)
library(palmerpenguins)
library(janitor)

lookup <- list("penguins_raw (penguins)" = "penguins_raw")
ignored <- c("sim1", "sim2", "sim3", "sim4", "table1", "table2", "table3", "table4a", "table4b", "table5", "Animals", "Oats", "Muscle", "Melanoma")

results <- as.data.frame(data(package = .packages(all.available = TRUE))$results)
for (i in 1:nrow(results)) {
    row <- results[i,]
    package <- row$Package

    library(package, warn.conflicts = FALSE, character.only = TRUE)

    item <- row$Item
    name <- lookup[[item]]
    if (is.null(name)) {
        name <- item
    }

    data <- tryCatch(get(name), error = function (cond) return(NULL))

    name <- str_replace_all(name, "\\.", "_")

    if (!(name %in% ignored) && !is.null(data) && is.data.frame(data)) {
        basename <- paste(name, "csv.gz", sep = ".")
        file <- paste("ibis/examples/data", basename, sep = "/")

        clean_data <- clean_names(data)
        write_csv(clean_data, file = file, quote = "needed", na = "")

        # write a column-name-uncleansed file if the clean names differ
        if (any(names(clean_data) != names(data))) {
            raw_basename <- paste(paste(name, "raw", sep = "_"), "csv.gz", sep = ".")
            raw_file <- paste("ibis/examples/data", raw_basename, sep = "/")
            write_csv(data, file = raw_file, quote = "needed", na = "")
        }

        text <- row$Title
        cat(text, file = paste("ibis/examples/descriptions", name, sep = "/"))
    }
}
