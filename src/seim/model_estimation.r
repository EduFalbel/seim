print(file.path(R.home("bin"), "R"))

library(sf)
library(spdep)
library(spflow)
library(readr)
# library(tidyverse)
library(MASS)
# library(rgdal)

options(warn=-1)
set.seed(42)

# Default weights matrix is row-normalized without distance decay based on queen contiguity neighborhood
default_wm_options <- function() {
    wm_options <- c(
        "neighbour" = "knn", # Either contiguity or knn
        "queen/k" = 8, # Either TRUE/FALSE or integer between 1 and n-1
        "weight" = "standard", # standard or dist_decay
        "style" = "W", # B W
        "type" = "idw"
    )
    return(wm_options)
}

create_weights_matrix <- function(node_data, options) {
    if (as.character(options[["neighbour"]]) == "contiguity") {
        neighbours <- spdep::poly2nb(node_data, queen = options[["queen/k"]])
    } else if (as.character(options[["neighbour"]] == "knn")) {
        neighbours <- knn2nb(knearneigh(st_centroid(node_data), k=as.numeric(options[["queen/k"]])))
    }

    if (as.character(options[["weight"]] == "dist_decay")) {
        weights_matrix <- listw2mat(nb2listwdist(neighbours, st_centroid(node_data), type = as.character(options[["type"]]), style = as.character(options[["style"]])))

    } else if (as.character(options[["weight"]] == "standard")) {
        weights_matrix <- nb2mat(neighbours, style = as.character(options[["style"]]), zero.policy = TRUE)#FALSE)

    }

    return(weights_matrix)
}

create_and_write_weights_matrix <- function(node_path, data_save_path, wm_options = NULL, ...) {

    node_data <- sf::st_read(node_path)
    if (is.null(wm_options)) {
        wm_options <- default_wm_options()
    }
    weights_matrix <- create_weights_matrix(node_data, wm_options)
    write.matrix(weights_matrix, file = paste0(data_save_path, "weights.txt"))
    return(weights_matrix)
}

read_data <- function(node_path, pair_path) {
    node_data <- sf::st_read(node_path)

    print(class(node_data))
    
    pair_data <- read_csv(pair_path)

    return(list(node_data, pair_data))
}

write_data <- function(node_data, pair_data, data_save_path) {
    print(data_save_path)

    write.csv(st_drop_geometry(node_data), file = paste0(data_save_path, "node_r.csv"), row.names = FALSE)
    write.csv(pair_data, file = paste0(data_save_path, "pair_r.csv"), row.names = FALSE)
}

read_and_write_data <- function(node_path, pair_path, data_save_path) {
    data <- read_data(node_path, pair_path)
    write_data(data[[1]], data[[2]], data_save_path)
}

create_cntrl <- function(cntrl_type) {
    if (cntrl_type == "SLA") {
       cntrl <- spflow_control(
        estimation_method = "mle",
        sdm_variables = "none"
       )
    }
    if (cntrl_type == "Aspatial") {
       cntrl <- spflow_control(
        estimation_method = "mle",
        sdm_variables = "none",
        model = "model_1"
       )
    }
    if (cntrl_type == "SLX") {
       cntrl <- spflow_control(
        sdm_variables = "all",
        model = "model_1"
       )
    }
    if (cntrl_type == "SDM") {
       cntrl <- spflow_control(
        sdm_variables = "all"
       )
    }

    return(cntrl)
}

estimate_model_params <- function(node_path, pair_path, dependent_var, wm_options, data_save_path = "/tmp/", region_name = "region", node_key = "ID", pair_orig_key = "ID_ORIG", pair_dest_key = "ID_DEST", cntrl_type = "SLA", form = NULL) {

    print("In estimate_model_params")

    if (is.null(form)) {
        form <- as.formula(paste0(dependent_var, " ~ ."))
    } else {
        # http://www.cookbook-r.com/Formulas/Creating_a_formula_from_a_string/
        form <- as.formula(form)
    }

    print("Built formula")

    # print(form)

    if (is.null(wm_options)) {
        wm_options <- default_wm_options()
    }

    print("Got weights matrix options")
    print(wm_options)
    print(class(wm_options))

    cntrl <- create_cntrl(cntrl_type)

    print("Built flow_cntrl")

    data <- read_data(node_path, pair_path)
    print("Read data")
    node <- data[[1]]
    pair <- data[[2]]

    print(class(node))

    weights_matrix <- create_weights_matrix(node, wm_options)

    print("Built weights matrix")

    print(class(node))
    print(class(pair))

    print(wm_options[["queen/k"]])



    net_nodes <- spflow::sp_network_nodes(
        network_id = region_name,
        node_neighborhood = weights_matrix,
        node_data = st_drop_geometry(node),
        node_key_column = node_key
    )

    net_pairs <- spflow::sp_network_pair(
        orig_net_id = region_name,
        dest_net_id = region_name,
        pair_data = pair,
        orig_key_column = pair_orig_key,
        dest_key_column = pair_dest_key
    )

    multi_net <- spflow::sp_multi_network(net_nodes, net_pairs)

    model <- spflow(
        flow_formula = form,
        sp_multi_network = multi_net,
        flow_control = cntrl
    )

    return(coef(model))

}