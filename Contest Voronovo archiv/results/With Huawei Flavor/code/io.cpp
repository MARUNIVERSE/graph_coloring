#include "io.h"

#include <algorithm>
#include <utility>
#include <set>
using namespace std;

Data read_data(string path_nodes, string path_links, string path_routing) {
    Data data;
    ifstream node_in(path_nodes), link_in(path_links), route_in(path_routing);

    std::string line;
    node_in >> line;
    route_in >> line;
    link_in >> line;

    // Read nodes.
    Node node  = {-1};
    while (node_in >> line) {
        auto tokens = tokenize(line, ',');
        node.uid_node = std::stoi(tokens[0]);
        data.nodes.push_back(node);
        data.node_uid2obj[node.uid_node] = node;
    }

    // Read links.
    Link link = {-1, -1, -1};
    while (link_in >> line) {
        auto tokens = tokenize(line, ',');

        link.uid_from = std::stoi(tokens[1]);
        link.uid_to = std::stoi(tokens[2]);
        link.uid_link = std::stoi(tokens[0]);

        data.links.push_back(link);
        data.link_uid2obj[link.uid_link] = link;
    }

    // Read paths.

    unordered_map<int, int> uid2path;
    while (route_in >> line) {
        auto tokens = tokenize(line, ',');
        int uid_path = std::stoi(tokens[0]);
        int ind = data.paths.size();
        int width = std::stoi(tokens[2]);
        if (uid2path.find(uid_path) != uid2path.end()) {
            ind = uid2path[uid_path];
        } else {
            Path path = {vector<Link>(), width, uid_path, 0};
            data.paths.push_back(path);
        }
        int link_uid = std::stoi(tokens[1]);
        Link lnk = data.link_uid2obj[link_uid];
        data.paths[ind].links.push_back(lnk);
        data.path_uid2obj[uid_path] = data.paths[ind];
    }
    return data;
}


vector<string> tokenize(string str, char delim) {
    stringstream in(str);
    vector<string> answer;
    string item;

    while (getline(in, item, delim)) {
        if (item.length()) {
            answer.push_back(item);
        }
    }
    return answer;
}

void write_data(const vector<OutEntry>& ans, std::string coloring_path,
        const Data& data) {
    ofstream out(coloring_path);
    out << "path_id,min_slice" << endl;
    set<int> tracked_ids;
    for (const auto& item : ans) {
        out << item.uid_path << "," << item.L_path << endl;
        tracked_ids.insert(item.uid_path);
    }
    for (const auto& item : data.paths) {
        auto item_id = item.uid_path;
        if (tracked_ids.find(item_id) == tracked_ids.end()) {
            out << item_id << "," << -1 << endl;
        }
    }
}
