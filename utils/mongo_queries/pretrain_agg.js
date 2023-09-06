/*
*
*   Aggregation pipeline to create pretraining dataset where
*   each statement has the corresponding graph data
*/

// use hol_step

db = connect('mongodb://localhost/hol_step');

db.train_val_test_data.aggregate([
    {
        $lookup: {
            from: "expression_graphs",
            localField: "conj",
            foreignField: "_id",
            as: "conj_graph"
        }
    },
    {
      $replaceRoot: {newRoot: {$mergeObjects: [{_id: "$_id", conj_graph:{
          $arrayElemAt: ["$conj_graph.graph", 0]}, conj: "$conj", stmt: "$stmt", split: "$split", y: "$y"}]}}
    },
    {
        $lookup: {
            from: "expression_graphs",
            localField: "stmt",
            foreignField: "_id",
            as: "stmt_graph"
        }
    },
    {
        $replaceRoot: {newRoot: {$mergeObjects: [{_id: "$_id", conj_graph: "$conj_graph", stmt_graph:{
                      $arrayElemAt: ["$stmt_graph.graph", 0]},
                        conj: "$conj", stmt: "$stmt", split: "$split", y: "$y"}]}}
    },

    {$out : "pretrain_graphs"}

])

// Set random index, and create index on it for fast sort

db.pretrain_graphs.updateMany(
    {rand_idx: {$exists: false} },
    [{$set:
            {rand_idx: {
                $function: {
                    body: function() {return Math.random();},
                    args: [],
                    lang: "js"
                }
                }}
    }]
    )

// Speedup from ~30s to 0.1s
db.pretrain_graphs.createIndex({rand_idx : 1})

