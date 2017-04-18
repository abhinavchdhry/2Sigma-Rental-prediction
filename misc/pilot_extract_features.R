############################
Import Data
############################

library(jsonlite)
setwd("/Users/zubin/Documents/NCSU/Courses/CSC 591 - BI/Capstone/")
data <- read_json("train.json", simplifyVector = TRUE)

############################
Create Data Frame
############################

df <- unlist(data$bathrooms)
names <- names(data)
for (i in 2:length(names)) {
  df <- cbind(df, data[names[i]][[1]])
}
df <- as.data.frame(df)
names(df) <- names

############################
Get strings for pet friendly
############################
cat_list = c()
dog_list = c()
pet_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("cats ",str,ignore.case = TRUE) || grepl("cat ",str,ignore.case = TRUE)) {
            cat_list <- c(cat_list,str)
        }
        if(grepl("dogs ",str,ignore.case = TRUE) || grepl("dog ",str,ignore.case = TRUE)) {
            dog_list <- c(dog_list,str)
        }
        if(grepl("pets ",str,ignore.case = TRUE) || grepl("pet ",str,ignore.case = TRUE)) {
            pet_list <- c(pet_list,str)
        }
    }
}
print(unique(cat_list))
print(unique(dog_list))
print(unique(pet_list))

############################
Get strings for elevator
############################
elevator_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("elevator",str,ignore.case = TRUE)) {
            elevator_list <- c(elevator_list,str)
        }
    }
}
print(unique(elevator_list))

############################
Get strings for fitness center
############################
fitness_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("fitness center",str,ignore.case = TRUE) || grepl("fitness centre",str,ignore.case = TRUE) || grepl("fitness ",str,ignore.case = TRUE) || grepl("yoga ",str,ignore.case = TRUE) || grepl("gym ",str,ignore.case = TRUE) || grepl("exercise ",str,ignore.case = TRUE) || grepl("aerobic ",str,ignore.case = TRUE) || grepl("cardio ",str,ignore.case = TRUE) || grepl("basketball ",str,ignore.case = TRUE)) {
            fitness_list <- c(fitness_list,str)
        }
    }
}
print(unique(fitness_list))

############################
Get strings for pool
############################
pool_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("swimming",str,ignore.case = TRUE) || grepl("pool",str,ignore.case = TRUE)) {
            pool_list <- c(pool_list,str)
        }
    }
}
print(unique(pool_list))

############################
Get strings for access list
############################
access_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("wheelchair",str,ignore.case = TRUE)) {
            access_list <- c(access_list,str)
        }
    }
}
print(unique(access_list))

############################
Get strings for concessions
############################
concessions_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("month .*free",str,ignore.case = TRUE) || grepl("free *.month",str,ignore.case = TRUE)) {
            concessions_list <- c(concessions_list,str)
        }
    }
}
print(unique(concessions_list))

############################
Get strings for Broker Fee
############################
fee_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("fee",str,ignore.case = TRUE)) {
            fee_list <- c(fee_list,str)
        }
    }
}
print(unique(fee_list))

############################
Get strings for Parking
############################
parking_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("parking",str,ignore.case = TRUE)) {
            parking_list <- c(parking_list,str)
        }
    }
}
print(unique(parking_list))

############################
Get strings for Storage
############################
storage_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("storage",str,ignore.case = TRUE) || grepl("garage",str,ignore.case = TRUE)) {
            storage_list <- c(storage_list,str)
        }
    }
}
print(unique(storage_list))

############################
Get strings for playroom
############################
playroom_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("playroom",str,ignore.case = TRUE) || grepl("play ",str,ignore.case = TRUE) || grepl("children ",str,ignore.case = TRUE) || grepl("nursery ",str,ignore.case = TRUE)) {
            playroom_list <- c(playroom_list,str)
        }
    }
}
print(unique(playroom_list))

############################
Get strings for connectivity
############################
connectivity_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("internet",str,ignore.case = TRUE) || grepl("wifi",str,ignore.case = TRUE) || grepl("wi-fi",str,ignore.case = TRUE) || grepl("cable",str,ignore.case = TRUE) || grepl("satellite",str,ignore.case = TRUE)) {
            connectivity_list <- c(connectivity_list,str)
        }
    }
}
print(unique(connectivity_list))

############################
Get strings for balcony
############################
balcony_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("balcony",str,ignore.case = TRUE) || grepl("terrace",str,ignore.case = TRUE)) {
            balcony_list <- c(balcony_list,str)
        }
    }
}
print(unique(balcony_list))

############################
Get strings for outdoor
############################
outdoor_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("outdoor.*space",str,ignore.case = TRUE) || grepl("bbq",str,ignore.case = TRUE) || grepl("grill",str,ignore.case = TRUE)) {
            outdoor_list <- c(outdoor_list,str)
        }
    }
}
print(unique(outdoor_list))

############################
Get strings for laundry
############################
laundry_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("washer",str,ignore.case = TRUE) || grepl("dryer",str,ignore.case = TRUE) || grepl("laundry",str,ignore.case = TRUE)) {
            laundry_list <- c(laundry_list,str)
        }
    }
}
print(unique(laundry_list))

############################
Get strings for dishwasher
############################
dishwasher_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("dish.*washer",str,ignore.case = TRUE)) {
            dishwasher_list <- c(dishwasher_list,str)
        }
    }
}
print(unique(dishwasher_list))

############################
Get strings for type of building
############################
building_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("building",str,ignore.case = TRUE)) {
            building_list <- c(building_list,str)
        }
    }
}
print(unique(building_list))

############################
Get strings for type of floor
############################
floor_list = c()
for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
        if(grepl("floor",str,ignore.case = TRUE)) {
            floor_list <- c(floor_list,str)
        }
    }
}
print(unique(floor_list))


LIST OF USABLE FEATURES
-----------------------
Pet Friendly? - Dogs and Cats, Dogs, Cats, Not Allowed, Allowed on Approval, Small pets (Some restriction)
Elevator 
Fitness Center / Yoga Room / Gym / Basketball Court
Swimming Pool 
Wheelchair Access 
Concessions (Free 1 month on annual lease)
Broker Fee (Reduced fee, no broker fees, half month, month, $250)
Parking Space 
Storage / Garage
Childrens room / Playroom 
High Speed Internet / Wifi Access / Cable / Satellite TV 
Balcony / Terrace
Outdoor Space (Common or Otherwise, BBQ, Grill)
Laundry  - Washer / Dryer in Unit, Laundry in Floor/Unit, Laundry in Building
Dishwasher
Type of building - Duplex, Single Storey, Lowrise, Studio
Type of Floor - Simple, Hardwood
Garden 
View 
Bedroom size - King, Queen (size of largest bedroom)
Housekeeping / Supervisor / Door Man
Sauna
Library
Bike Storage
Bathrooms - 1.5, 2 
ATM Machine 

FEATURE TITLES
--------------
pets
elevator
fitness
pool
accessibility
concessions
broker_fee
parking
storage
playroom
connectivity
balcony
outdoor
laundry
dishwasher
building
floor
garden
view
bedroom
services
sauna
library
bike_storage
bathrooms
bathroom
atm
