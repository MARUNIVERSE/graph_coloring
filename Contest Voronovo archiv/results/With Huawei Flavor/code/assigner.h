#pragma once

#include "interfaces.h"
#include "common.h"

#include <algorithm>
#include <queue>
#include <exception>
using namespace std;


enum class ScoreType {
    WIDTH,
    ADJACENT,
    MAX_LOAD,
    AVG_LOAD,
    SUM_LOAD,
    MAX_CNT,
    AVG_CNT,
    SUM_CNT
};


class GreedyAssigner : public IAssigner {
public:
    GreedyAssigner(ScoreType scorer) : scorer(scorer) {}

    void setMetric(ScoreType aScorer) { scorer = aScorer; }

    virtual vector<OutEntry> assign(const Data& aData) override {
        Answer answer(aData);
        Data data = aData;

        setScores(data);

        sort(data.paths.begin(), data.paths.end(),
                [&](const Path& lhs, const Path& rhs) -> bool {
            return lhs.score < rhs.score;
        });

        for (const auto& path : data.paths) {
            auto place = answer.findAdd(path.uid_path);
            if (place != -1) {
                answer.add(path.uid_path, place);
            }
        }
        return answer.getResponse();
    }

    void setScores(Data& data) {
        if (scorer == ScoreType::WIDTH) {
            setWidthScores(data);
        } else if (scorer == ScoreType::ADJACENT) {
            setAdjacentScores(data);
        } else if (scorer == ScoreType::MAX_LOAD) {
            setMaxLoadScores(data);
        } else if (scorer == ScoreType::AVG_LOAD) {
            setAvgLoadScores(data);
        } else if (scorer == ScoreType::SUM_LOAD) {
            setSumLoadScores(data);
        } else if (scorer == ScoreType::MAX_CNT) {
            setMaxCntScores(data);
        } else if (scorer == ScoreType::AVG_CNT) {
            setAvgCntScores(data);
	} else if (scorer == ScoreType::SUM_CNT) {
            setSumCntScores(data);
        }
    }

    // Good when small.
    void setWidthScores(Data& data, int scale=1) {
        for (auto& path : data.paths) {
            path.score = path.width * scale;
        }
    }

    // Good when small.
    void setAdjacentScores(Data& data, int scale=1) {
        for (auto& path : data.paths) {
            path.score = data.adjacent_paths[path.uid_path].size() * scale;
        }
    }

    // Good when small.
    void setMaxLoadScores(Data& data, int scale=1) {
        for (auto& path : data.paths) {
            for (const auto& link : path.links) {
                 int uid_link = link.uid_link;
                 int demand = data.link_uid2demand[uid_link];
                 path.score = std::max<double>(demand, path.score);
            }
            path.score *= scale;
        }
    }

    // Good when small.
    void setMaxCntScores(Data& data, int scale=1) {
        for (auto& path : data.paths) {
            for (const auto& link : path.links) {
                int uid_link = link.uid_link;
                int count = data.link_uid2cnt[uid_link];
                path.score = std::max<double>(path.score, count);
            }
            path.score *= scale;
        }
    }

    // Good when small.
    void setSumCntScores(Data& data, int scale=1) {
        for (auto& path : data.paths) {
            for (const auto& link : path.links) {
                int uid_link = link.uid_link;
                int count = data.link_uid2cnt[uid_link];
                path.score += count;
            }
            path.score *= scale;
        }
    }

    // Good when small.
    void setAvgCntScores(Data& data, int scale=1) {
        for (auto& path : data.paths) {
            for (const auto& link : path.links) {
                int uid_link = link.uid_link;
                int count = data.link_uid2cnt[uid_link];
                path.score += count;
            }
            path.score *= scale;
	    path.score /= path.links.size();
        }
    }

    // Good when small
    void setAvgLoadScores(Data& data, int scale=1) {
	for (auto& path: data.paths) {
            double load = 0;
	    for (const auto& link: path.links) {
                int uid_link = link.uid_link;
                load += data.link_uid2demand[uid_link];
            }
	    load /= path.links.size();
	    path.score = load * scale;
	}
    }

    // Good when small
    void setSumLoadScores(Data& data, int scale=1) {
	for (auto& path : data.paths) {
	    double load = 0;
	    for (const auto& link : path.links) {
         	int uid_link = link.uid_link;
                load += data.link_uid2demand[uid_link];
	    }
	    path.score = load * scale;
	}
    }

private:
    ScoreType scorer;
};

bool operator<(const Path& lhs, const Path& rhs) {
    return lhs.score < rhs.score;
}

enum class PriorityType {
    AVG_FREE,
    AVG_IMPACT,
    MAX_IMPACT
};

class PriorityAssigner : public IAssigner {
public:
    PriorityAssigner(PriorityType scorer) : scorer(scorer) {}
    virtual vector<OutEntry> assign(const Data& aData) override {
        Answer answer(aData);
        Data data = aData;
        initStruct(data);
        priority_queue<Path> queue;
        int stamp = 0;
        for (auto path : data.paths) {
            path.score = computeScore(path, data, scorer);
            path.stamp = stamp;
            uid_path2stamp[path.uid_path] = stamp;
            queue.push(path);
	    ++stamp;
        }
        set<int> path_out;
        while (queue.size()) {
            auto head = queue.top();
            queue.pop();
            if (head.stamp != uid_path2stamp[head.uid_path]) continue;
            ++stamp;
            path_out.insert(head.uid_path);
            auto place = answer.findAdd(head.uid_path);
            if (place != -1) {
                answer.add(head.uid_path, place);
                for (const auto& lnk : head.links) {
		    assert(uid_lnk2cap.find(lnk.uid_link) != uid_lnk2cap.end());
                    uid_lnk2cap[lnk.uid_link] -= head.width;
                    assert(uid_lnk2cap[lnk.uid_link] >= 0);
                }
                for (auto ngh_uid : data.adjacent_paths[head.uid_path]) {
                    if (path_out.find(ngh_uid) != path_out.end()) {
                        continue;
                    }
                    auto path = data.path_uid2obj[ngh_uid];
                    path.score = computeScore(path, data, scorer);
                    path.stamp = stamp;
                    uid_path2stamp[path.uid_path] = stamp;
                    queue.push(path);
		    ++stamp;
                }
            }
        }
        return answer.getResponse();
    }
private:
    void initStruct(const Data& data) {
        uid_lnk2cap.clear();
        uid_path2stamp.clear();
        for (const auto& lnk : data.links) {
            uid_lnk2cap[lnk.uid_link] = data.BW;
        }
    }

    double computeScore(const Path& path, const Data& data, PriorityType type) {
        if (type == PriorityType::AVG_FREE) {
            return computeAvgFreeScore(path, data);
        } else if (type == PriorityType::AVG_IMPACT) {
            return computeAvgImpactScore(path, data);
        } else if (type == PriorityType::MAX_IMPACT) {
            return computeMaxImpactScore(path, data);
        } else {
            throw std::logic_error("Unknown scorer!");
        }
    }

    // Big is good.
    double computeAvgFreeScore(const Path& path, const Data& aData) {
        double sum = 0;
        for (const auto& lnk : path.links) {
            sum += uid_lnk2cap[lnk.uid_link];
        }
        return sum / path.links.size();
    }

    // Big is good.
    double computeAvgImpactScore(const Path& path, const Data& aData) {
        double avgFree = computeAvgFreeScore(path, aData);
        return  -path.width / avgFree;
    }

    // Big is good.
    double computeMaxImpactScore(const Path& path, const Data& aData) {
        double impact  = 0;
        for (const auto& lnk : path.links) {
            impact = std::max<double>(impact, path.width / uid_lnk2cap[lnk.uid_link]);
        }
        return -impact;
    }
private:
    unordered_map<int, int> uid_lnk2cap;
    unordered_map<int, int> uid_path2stamp;
    PriorityType scorer;
};


class GurobiAssigner : public IAssigner {
public:
    virtual vector<OutEntry> assign(const Data& data) override = 0;
};

