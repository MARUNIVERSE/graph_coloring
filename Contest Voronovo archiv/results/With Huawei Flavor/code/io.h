#pragma once

#include <cstdio>
#include <sstream>
#include <fstream>
#include <iostream>

#include "common.h"


vector<string> tokenize(string str, char delim);
Data read_data(string path_nodes, string path_links, string path_routing);
void write_data(const vector<OutEntry>& ans, std::string coloring_path,
        const Data& data);

