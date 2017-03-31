install.packages("rjson")
library(rjson)

data <- rjson::fromJSON(paste(readLines("train_with_features.json")))

attrs <- c("pets", "elevator", "fitness", "pool", "accessibility", "concessions", "broker_fee", "parking", "storage", "playroom", "connectivity", "balcony", "outdoor", "laundry", "dishwasher", "building", "floor", "garden", "view", "bedroom", "services", "sauna", "library", "bike_storage", "bathroom", "atm")

df <- c()
for (i in 1:length(data)) {
  print(i)
  current_row <- data[i][[1]]
  vec <- c()
  
  for (attr in attrs) {
    if (attr %in% names(current_row)) {
      # pets, broker_fee, building, floor, bathroom
      if (attr %in% c("pets", "broker_fee", "building", "floor", "bathroom")) {
        vec <- c(vec, as.character(current_row[attr]))
      }
      else {
        vec <- c(vec, as.numeric(current_row[attr]))
      }
    }
    else {
      vec <- c(vec, 0)
    }
  }
  
  df <- rbind(df, vec)
}

