library(sf)
library(spdep)
library(spflow)
library(readr)
# library(tidyverse)
library(MASS)
# library(rgdal)


set.seed(42)

create_weights_matrix <- function(options, node_data) {
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

read_data <- function(node_path, pair_path, wm_options) {
    node_data <- sf::st_read(node_path)
    pair_data <- read_csv(pair_path)

    weights_matrix <- create_weights_matrix(wm_options, node_data)

    # print(class(node_data))

    return(list(node_data, pair_data, weights_matrix))
}

write_data <- function(node_data, pair_data, weights_matrix, data_save_dir) {
    write.csv(st_drop_geometry(node_data), file = paste0(data_save_dir, "node.csv"), row.names = FALSE)
    write.csv(pair_data, file = paste0(data_save_dir, "pair.csv"), row.names = FALSE)
    write.matrix(weights_matrix, file = paste0(data_save_dir, "weights.txt"))
}

read_and_write_data <- function(node_path, pair_path, wm_options, data_save_dir) {
    data <- read_data(node_path, pair_path, wm_options)
    write_data(data[[1]], data[[2]], data[[3]], data_save_dir)
    return(data)
}



create_cntrl <- function(cntrl_type) {
    if (cntrl_type == 'SLA') {
       cntrl <- spflow_control(
        sdm_variables = "none"
       )
    }
    if (cntrl_type == 'Aspatial') {
       cntrl <- spflow_control(
        sdm_variables = "none",
        model = "model_1"
       )
    }
    if (cntrl_type == 'SLX') {
       cntrl <- spflow_control(
        sdm_variables = "all",
        model = "model_1"
       )
    }
    if (cntrl_type == 'SLA') {
       cntrl <- spflow_control(
        sdm_variables = "all"
       )
    }
    
    return(cntrl)
}

estimate_model_params <- function(node_path, pair_path, data_save_path, region_name, dependent_var, wm_options, node_key = "ID", pair_orig_key = "ID_ORIG", pair_dest_key = "ID_DEST", form = NULL, cntrl_type = NULL) {


    if(is.null(form)) {
        form <- as.formula(paste0(dependent_var, " ~ ."))
    } else {
        # http://www.cookbook-r.com/Formulas/Creating_a_formula_from_a_string/
        form <- as.formula(form)
    }

    print(form)

    if(is.null(cntrl_type)) {
        # Defaults to SLA model
        cntrl <- spflow_control(
            sdm_variables = "none"
        )
    } else {
        cntrl <- create_cntrl(cntrl_type)
    }

    data <- read_and_write_data(node_path, pair_path, wm_options, data_save_path)
    print("Out")
    node <- data[[1]]
    pair <- data[[2]]
    weights_matrix <- data[[3]]


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
        sp_multi_network =  multi_net,
        flow_control = cntrl
    )

    return(coef(model))

}