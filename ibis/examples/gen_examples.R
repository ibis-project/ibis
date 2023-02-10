library(tidyverse)
library(stringr)
library(palmerpenguins)
library(janitor)

lookup <- list("penguins_raw (penguins)" = "penguins_raw")

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

    if (!is.null(data) && is.data.frame(data)) {
        basename <- paste(name, "csv.gz", sep = ".")
        file <- paste("ibis/examples/data", basename, sep = "/")
        write_csv(clean_names(data), file = file, quote = "needed", na = "")
        text <- row$Title
        cat(text, file = paste("ibis/examples/descriptions", name, sep = "/"))
    }
}
