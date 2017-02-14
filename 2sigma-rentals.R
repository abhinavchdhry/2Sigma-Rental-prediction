#install.packages("jsonlite")
#install.packages("dplyr")
#install.packages("ggplot2")
#install.packages("ggmap")

library(jsonlite)
library(dplyr)
library(ggplot2)
library(ggmap)

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

# Plot map of locations
# Extract (long, lat) location vectors
locations <- cbind(unlist(df$longitude), unlist(df$latitude))
locations <- as.data.frame(locations)
names(locations) <- c("Long", "Lat")
qmplot(Long, Lat, data=locations, colour=I('red'), size=I(1))
#qmplot(Long, Lat, data=locations, colour=I('red'), size=I(5), source="google", maptype="roadmap")

## Notice some outliers near Africa and on the U.S. West coast, and some in the central U.S.
## Restrict locations to near NY using clustering
## Let K-mean extract the NY cluster

