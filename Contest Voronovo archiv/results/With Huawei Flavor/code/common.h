#pragma once

#include <vector>
#include <set>
#include <cassert>
#include <algorithm>
#include <unordered_map>

using namespace std;

struct Node {
    int uid_node;
};

struct Link {
    int uid_from;
    int uid_to;
    int uid_link;
};

struct Path {
// Main entries.
    vector<Link> links;
    int width;
    int uid_path;

// Temporary entries.
    double score;
    int stamp;
};

struct Data {
// Main Entries.
    int BW = 320;
    int shift = 0;

    vector<Node> nodes;
    vector<Link> links;
    vector<Path> paths;

    unordered_map<int, Node> node_uid2obj;
    unordered_map<int, Path> path_uid2obj;
    unordered_map<int, Link> link_uid2obj;

// Structural entries.
    unordered_map<int, vector<Path>> link_uid2paths;
    unordered_map<int, int> link_uid2demand;
    unordered_map<int, int> link_uid2cnt;
    unordered_map<int, set<int>> adjacent_paths;

    void initStructure() {
        link_uid2paths.clear();
        link_uid2demand.clear();
        adjacent_paths.clear();
        link_uid2cnt.clear();

        for (const auto& path : paths) {
            for (const auto& link : path.links) {
                link_uid2paths[link.uid_link].push_back(path);
                link_uid2demand[link.uid_link] += path.width;
                link_uid2cnt[link.uid_link] += 1;
            }
        }
        for (const auto& data : link_uid2paths) {
            const auto& paths = data.second;
            for (int idx = 0; idx < int(paths.size()); ++idx) {
                auto uid_first = paths[idx].uid_path;
                for (int jdx = idx + 1; jdx < int(paths.size()); ++jdx) {
                    auto uid_second = paths[idx].uid_path;
                    adjacent_paths[uid_first].insert(uid_second);
                    adjacent_paths[uid_second].insert(uid_first);
                }
            }
        }
    }

    bool adjacent(int lhs, int rhs) {
        return (adjacent_paths[lhs].find(rhs) != adjacent_paths[lhs].end());
    }
};


struct OutEntry {
    int uid_path;
    int L_path, R_path;
};


class Answer {
public:
    Answer(const Data& data) : data(data) {
        vector<int> zeros(data.BW);
        for (const auto& link : data.links) {
            used[link.uid_link] = zeros;
        }
    }

    bool canAdd(int uidPath, int L) {
        auto& path = data.path_uid2obj.at(uidPath);
        for (const auto& link : path.links) {
            for (int ind = L; ind < L + path.width; ++ind) {
                if (ind >= data.BW || used[link.uid_link][ind]) {
                    return false;
                }
            }
        }
        return true;
    }

    int findAdd(int uidPath) {
        vector<int> positions(data.BW);
        auto& path = data.path_uid2obj.at(uidPath);
        for (const auto& link : path.links) {
            for (int ind = 0; ind < data.BW; ++ind) {
                positions[ind] += used[link.uid_link][ind];
            }
        }
        vector<int> lengths;
        vector<pair<int, int>> lengthPlace;
        int tail = 0;
        for (int idx = 0; idx < int(positions.size()); ++idx) {
            if (positions[idx] == 0) {
                ++tail;
            } else  {
                tail = 0;
            }
            lengths.push_back(tail);
            lengthPlace.emplace_back(tail, idx);
	    assert(tail <= data.BW);
        }
        sort(lengthPlace.begin(), lengthPlace.end());
        for (const auto& item : lengthPlace) {
            if (item.first >= path.width) {
		assert(canAdd(uidPath, item.second - path.width + 1));
		assert(item.second - path.width + 1 >= 0);
                return item.second - path.width + 1;
            }
        }
        return -1;
    }

    void add(int uidPath, int L, bool check=true) {
        if (check) {
            assert(canAdd(uidPath, L));
        }
        if (added.find(uidPath) == added.end()) {
            auto& path = data.path_uid2obj.at(uidPath);

            added.insert(uidPath);
            OutEntry item;
            item.uid_path = uidPath;
            item.L_path = L;
            item.R_path = L + path.width;
            response.push_back(item);

            for (const auto& link : path.links) {
                for (int ind = item.L_path; ind < item.R_path; ++ind) {
                    assert(ind < data.BW);
		    assert(used[link.uid_link][ind] == 0);
                    used[link.uid_link][ind] += 1;
                }
            }
        }
    }

    vector<OutEntry> getResponse() const {
        return response;
    }

private:
    const Data& data;
    vector<OutEntry> response;
    set<int> added;
    unordered_map<int, vector<int>> used;
};

