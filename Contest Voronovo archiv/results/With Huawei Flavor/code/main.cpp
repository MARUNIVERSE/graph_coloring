#include <vector>
#include <memory>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <unordered_map>
#include <sstream>
#include <iostream>
#include <istream>
#include <string>


#include "common.h"
#include "io.h"
#include "assigner.h"

using namespace std;

typedef long long ll;


vector<OutEntry> solveData(const Data& data) {
    vector<OutEntry> best;
    for (auto scorer : {ScoreType::WIDTH, ScoreType::ADJACENT,
            ScoreType::SUM_LOAD, ScoreType::AVG_LOAD,
            ScoreType::SUM_CNT, ScoreType::AVG_CNT,
            ScoreType::MAX_LOAD, ScoreType::MAX_CNT}) {
        std::shared_ptr<IAssigner> assigner =
            make_shared<GreedyAssigner>(scorer);
        auto result = assigner->assign(data);
        if (best.size() < result.size()) {
            best = result;
        }
    }
    for (auto scorer : {PriorityType::AVG_FREE,
            PriorityType::AVG_IMPACT,
            PriorityType::MAX_IMPACT}) {
        std::shared_ptr<IAssigner> assigner =
            make_shared<PriorityAssigner>(scorer);
        auto result = assigner->assign(data);
        if (best.size() < result.size()) {
            best = result;
        }
    }
    return best;
}


int main() {
    vector<string> directories;
    for (int ind = 49; ind <= 64; ++ind) {
        directories.push_back("../inputs/" + std::to_string(ind));
    }
    vector<string> files = {
        "/nodesinfo.csv", "/links.csv",
        "/newrouting.csv", "/coloring.csv"
    };
    for (const auto& dirPath : directories) {
        cout << "Started directory: " << dirPath << "." << endl;
        auto data = read_data(
                dirPath + files[0], dirPath + files[1], dirPath + files[2]);
        cout << "Data has: " << data.paths.size()
            << " paths, " << data.links.size()
            << " links, " << data.nodes.size() << " nodes." << endl;
        data.initStructure();
        auto best = solveData(data);
        write_data(best, dirPath + files[3], data);
        cout << "Processed directory: " << dirPath << "." << endl;
        cout << "Solution size is: " << best.size() << " from "
            << data.paths.size() << "." << endl;
    }
}
