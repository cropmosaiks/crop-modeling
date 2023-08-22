
r2_general <-function(actual, predictions) { 
  r2 <- 1 - sum((predictions - actual) ^ 2) / sum((actual - mean(actual))^2)
  return(r2)
}

r2_pears <- function(actual, predictions) { 
  r2 <- cor(actual, predictions) ^ 2
  return(r2)
}

stderror <- function(x) { 
  sd(x)/sqrt(length(x))
}
