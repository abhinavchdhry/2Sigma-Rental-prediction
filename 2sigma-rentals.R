#install.packages("jsonlite")
#install.packages("dplyr")

library(jsonlite)
library(dplyr)

setwd("/Users/achoudh3/Desktop/")
data <- read_json("train.json", simplifyVector = TRUE)

df <- unlist(data$bathrooms)
names <- names(data)
for (i in 2:length(names)) {
  df <- cbind(df, data[names[i]][[1]])
}
df <- as.data.frame(df)
names(df) <- names

# Split df into high, medium , low
df.high = df %>% filter(interest_level == "high")
df.medium = df %>% filter(interest_level == "medium")
df.low = df %>% filter(interest_level == "low")

# Feature listing
features = c()
for (i in 1:nrow(df)) {
  features.current = unlist(df[i,]$features)
  if (!is.null(features.current)) {
    features <- union(features, features.current)
  }
}

