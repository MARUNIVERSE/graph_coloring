#pragma once

#include "common.h"
#include <utility>

using namespace std;


class IAssigner {
public:
    virtual vector<OutEntry> assign(const Data& data)=0;
    virtual ~IAssigner(){};
};

class ISplitter {
public:
    virtual vector<Data> split(const Data& data)=0;
    virtual ~ISplitter(){};
};

class IMerger{
public:
    virtual vector<OutEntry> merge(
            const vector<vector<OutEntry>>& candidates,
            const Data& data)=0;
    virtual ~IMerger(){};
};


