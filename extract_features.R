library(jsonlite)
setwd("/Users/zubin/Documents/NCSU/Courses/CSC 591 - BI/Capstone/2Sigma-Rental-prediction-challenge/")
# data <- read_json("train.json", simplifyVector = TRUE)
data <- read_json("test.json", simplifyVector = TRUE)

df <- unlist(data$bathrooms)
names <- names(data)
for (i in 2:length(names)) {
  df <- cbind(df, data[names[i]][[1]])
}
df <- as.data.frame(df)
names(df) <- names

df[,"pets"] <- NULL
df[,"elevator"] <- NULL
df[,"fitness"] <- NULL
df[,"pool"] <- NULL
df[,"accessibility"] <- NULL
df[,"concessions"] <- NULL
df[,"broker_fee"] <- NULL
df[,"parking"] <- NULL
df[,"storage"] <- NULL
df[,"playroom"] <- NULL
df[,"connectivity"] <- NULL
df[,"balcony"] <- NULL
df[,"outdoor"] <- NULL
df[,"laundry"] <- NULL
df[,"dishwasher"] <- NULL
df[,"building"] <- NULL
df[,"floor"] <- NULL
df[,"garden"] <- NULL
df[,"view"] <- NULL
df[,"bedroom_size"] <- NULL
df[,"services"] <- NULL
df[,"sauna"] <- NULL
df[,"library"] <- NULL
df[,"bike_storage"] <- NULL
df[,"bathroom_type"] <- NULL
df[,"atm"] <- NULL

for (i in 1:nrow(df)) {
    for(str in df$features[[i]]) {
    	# PETS
        if(grepl("pets.*not.*allowed",str,ignore.case = TRUE)) {
            df[i,"pets"] <- "Pets Not Allowed"
        }
        else if(grepl("pets.*approval",str,ignore.case = TRUE) || grepl("pets.*case",str,ignore.case = TRUE)) {
            df[i,"pets"] <- "Pets On Approval"
        }
        else if(grepl("pets.*ok",str,ignore.case = TRUE) || grepl("pets.*okay",str,ignore.case = TRUE) || grepl("pets.*friendly",str,ignore.case = TRUE) || grepl("pet.*friendly",str,ignore.case = TRUE)) {
            df[i,"pets"] <- "Pets Allowed"
        }
        else if(grepl("dogs.*approval",str,ignore.case = TRUE) || grepl("dog.*approval",str,ignore.case = TRUE) || grepl("dogs.*case",str,ignore.case = TRUE) || grepl("specific.*dog",str,ignore.case = TRUE) || grepl("small.*dog",str,ignore.case = TRUE)) {
            df[i,"pets"] <- "Dogs on Approval"
        }
        else if(grepl("dog.*ok",str,ignore.case = TRUE) || grepl("dogs.*ok",str,ignore.case = TRUE)) {
            df[i,"pets"] <- "Dogs Allowed"
        }
        else if(grepl("cats ",str,ignore.case = TRUE) || grepl("cat ",str,ignore.case = TRUE)) {
            df[i,"pets"] <- "Cats Allowed"
        }

        # ELEVATOR
        if(grepl("elevator",str,ignore.case = TRUE)) {
            df[i,"elevator"] <- TRUE
        } else {
        	df[i,"elevator"] <- FALSE
        }

        # FITNESS CENTER
        if(grepl("fitness center",str,ignore.case = TRUE) || grepl("fitness centre",str,ignore.case = TRUE) || grepl("fitness ",str,ignore.case = TRUE) || grepl("yoga ",str,ignore.case = TRUE) || grepl("gym ",str,ignore.case = TRUE) || grepl("exercise ",str,ignore.case = TRUE) || grepl("aerobic ",str,ignore.case = TRUE) || grepl("cardio ",str,ignore.case = TRUE) || grepl("basketball ",str,ignore.case = TRUE)) {
            df[i,"fitness"] <- TRUE
        } else {
        	df[i,"fitness"] <- FALSE
        }

        # POOL
        if(grepl("swimming",str,ignore.case = TRUE) || grepl("pool",str,ignore.case = TRUE)) {
            df[i,"pool"] <- TRUE
        } else {
        	df[i,"pool"] <- FALSE
        }

        # ACCESSIBILITY
        if(grepl("wheelchair",str,ignore.case = TRUE)) {
            df[i,"accessibility"] <- TRUE
        } else {
        	df[i,"accessibility"] <- FALSE
        }

        # CONCESSIONS
        if(grepl("month .*free",str,ignore.case = TRUE) || grepl("free *.month",str,ignore.case = TRUE)) {
            df[i,"concessions"] <- TRUE
        } else {
        	df[i,"concessions"] <- FALSE
        }

        # BROKER FEE
        if(grepl("no fee",str,ignore.case = TRUE) || grepl("no.*broker.*fee",str,ignore.case = TRUE)) {
            df[i,"broker_fee"] <- "No Fee"
        } else if(grepl("reduced.*fee",str,ignore.case = TRUE) || grepl("low.*fee",str,ignore.case = TRUE)) {
        	df[i,"broker_fee"] <- "Reduced Fee"
        }

        # PARKING
        if(grepl("parking",str,ignore.case = TRUE)) {
            df[i,"parking"] <- TRUE
        } else {
        	df[i,"parking"] <- FALSE
        }

		# STORAGE
        if(grepl("storage",str,ignore.case = TRUE) || grepl("garage",str,ignore.case = TRUE)) {
            df[i,"storage"] <- TRUE
        } else {
        	df[i,"storage"] <- FALSE
        }

        # PLAYROOM
        if(grepl("playroom",str,ignore.case = TRUE) || grepl("play ",str,ignore.case = TRUE) || grepl("children ",str,ignore.case = TRUE) || grepl("nursery ",str,ignore.case = TRUE)) {
            df[i,"playroom"] <- TRUE
        } else {
        	df[i,"playroom"] <- FALSE
        }   

        # CONNECTIVITY
        if(grepl("internet",str,ignore.case = TRUE) || grepl("wifi",str,ignore.case = TRUE) || grepl("wi-fi",str,ignore.case = TRUE) || grepl("cable",str,ignore.case = TRUE) || grepl("satellite",str,ignore.case = TRUE)) {
            df[i,"connectivity"] <- TRUE
        } else {
        	df[i,"connectivity"] <- FALSE
        }   

        # BALCONY / TERRACE
        if(grepl("balcony",str,ignore.case = TRUE) || grepl("terrace",str,ignore.case = TRUE)) {
            df[i,"balcony"] <- TRUE
        } else {
        	df[i,"balcony"] <- FALSE
        }   

        # OUTDOOR SPACE / BBQ
        if(grepl("outdoor.*space",str,ignore.case = TRUE) || grepl("bbq",str,ignore.case = TRUE) || grepl("grill",str,ignore.case = TRUE)) {
            df[i,"outdoor"] <- TRUE
        } else {
        	df[i,"outdoor"] <- FALSE
        }  

        # Laundry
        if(grepl("washer.*dryer",str,ignore.case = TRUE) || grepl("laundry",str,ignore.case = TRUE)) {
            df[i,"laundry"] <- TRUE
        } else {
        	df[i,"laundry"] <- FALSE
        }   

        # DISHWASHER
        if(grepl("dish.*washer",str,ignore.case = TRUE)) {
            df[i,"dishwasher"] <- TRUE
        } else {
        	df[i,"dishwasher"] <- FALSE
        } 

        # TYPE OF BUILDING
        if(grepl("single.*storey",str,ignore.case = TRUE)) {
            df[i,"building"] <- "Single Storey"
        } else if(grepl("duplex",str,ignore.case = TRUE)) {
        	df[i,"building"] <- "Duplex"
        } else if(grepl("low.*rise",str,ignore.case = TRUE)) {
        	df[i,"building"] <- "Lowrise"
        } else if(grepl("studio",str,ignore.case = TRUE)) {
        	df[i,"building"] <- "Studio"
        } 

        # TYPE OF FLOOR 
        if(grepl("redwood",str,ignore.case = TRUE)) {
            df[i,"floor"] <- "Redwood"
        } else if(grepl("hardwood",str,ignore.case = TRUE)) {
        	df[i,"floor"] <- "Hardwood"
        } else if(grepl("oak.*floor",str,ignore.case = TRUE)) {
        	df[i,"floor"] <- "Oak"
        } else if(grepl("parquet",str,ignore.case = TRUE)) {
        	df[i,"floor"] <- "Parquet"
        } else if(grepl("herringbone",str,ignore.case = TRUE)) {
        	df[i,"floor"] <- "Herringbone"
        } 

        # GARDEN
        if(grepl("garden",str,ignore.case = TRUE)) {
            df[i,"garden"] <- TRUE
        } else {
        	df[i,"garden"] <- FALSE
        } 

        # VIEW
        if(grepl("view",str,ignore.case = TRUE) && grepl("interview",str,ignore.case = TRUE) && grepl("viewing",str,ignore.case = TRUE)) {
            df[i,"view"] <- TRUE
        } else {
        	df[i,"view"] <- FALSE
        } 

        # BEDROOM
        if(grepl("king.*bed",str,ignore.case = TRUE) || grepl("huge.*bed",str,ignore.case = TRUE)) {
            df[i,"bedroom_size"] <- "King"
        } else if(grepl("queen.*bed",str,ignore.case = TRUE)) {
        	df[i,"bedroom_size"] <- "Queen"
        } 

        # SERVICES - Housekeeping / Supervisor / Door Man
        if(grepl("housekeep",str,ignore.case = TRUE) || grepl("supervisor",str,ignore.case = TRUE) || grepl("door.*man",str,ignore.case = TRUE)) {
        	df[i,"services"] <- TRUE
        } else {
        	df[i,"services"] <- FALSE
        } 

        # SAUNULL
        if(grepl("sauna",str,ignore.case = TRUE)) {
            df[i,"sauna"] <- TRUE
        } else {
        	df[i,"sauna"] <- FALSE
        } 

        # LIBRARY
        if(grepl("library",str,ignore.case = TRUE) || grepl("reading",str,ignore.case = TRUE)) {
            df[i,"library"] <- TRUE
        } else {
        	df[i,"library"] <- FALSE
        } 

        # BIKE STORAGE
        if(grepl("bike.*storage",str,ignore.case = TRUE)) {
            df[i,"bike_storage"] <- TRUE
        } else {
        	df[i,"bike_storage"] <- FALSE
        } 

        # BATHROOMS
        # if(grepl("1.5.*bath",str,ignore.case = TRUE)) {
        #     df[i,"bathrooms"] <- "1.5"
        # }
        # else if(grepl("1/2.*bath",str,ignore.case = TRUE)) {
        #     df[i,"bathrooms"] <- "0.5"
        # }
        # else if(grepl("1.*bath",str,ignore.case = TRUE) || grepl("one.*bath",str,ignore.case = TRUE)) {
        #     df[i,"bathrooms"] <- "1"
        # } else if(grepl("2.5.*bath",str,ignore.case = TRUE)) {
        #     df[i,"bathrooms"] <- "2.5"
        # }
        # else if(grepl("2.*bath",str,ignore.case = TRUE) || grepl("two.*bath",str,ignore.case = TRUE)) {
        #     df[i,"bathrooms"] <- "2"
        # } else if(grepl("3.*bath",str,ignore.case = TRUE) || grepl("three.*bath",str,ignore.case = TRUE)) {
        #     df[i,"bathrooms"] <- "3"
        # } 

        # BATHROOM 
        if(grepl("marble.*bath",str,ignore.case = TRUE)) {
            df[i,"bathroom_type"] <- "Marble"
        } else if(grepl("granite.*bath",str,ignore.case = TRUE)) {
        	df[i,"bathroom_type"] <- "Granite"
        } else if(grepl("window.*bath",str,ignore.case = TRUE)) {
        	df[i,"bathroom_type"] <- "Window"
        } 

        # ATM
        if(grepl("atm",str,ignore.case = TRUE) && grepl("batman",str,ignore.case = TRUE) && grepl("treatment",str,ignore.case = TRUE)) {
            df[i,"atm"] <- TRUE
        } else {
        	df[i,"atm"] <- FALSE
        } 

    }
}


# DEMO 
# for (i in 1:nrow(df)) {
#     for(str in df$features[[i]]) {
#     	if(grepl("single.*storey",str,ignore.case = TRUE)) {
#             df[i,"building"] <- "Single Storey"
#         } else if(grepl("duplex",str,ignore.case = TRUE)) {
#         	df[i,"building"] <- "Duplex"
#         } else if(grepl("low.*rise",str,ignore.case = TRUE)) {
#         	df[i,"building"] <- "Lowrise"
#         } else if(grepl("studio",str,ignore.case = TRUE)) {
#         	df[i,"building"] <- "Studio"
#         } 
#     }
# }

# file_conn <- file("train_with_features.json","w")
file_conn <- file("test_with_features.json","w")
writeLines(toJSON(df),file_conn)
close(file_conn)
