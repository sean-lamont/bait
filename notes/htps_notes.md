
- Assuming all goals in tree are unique 
    - Multiple parents to a goal can exist in hypergraph, but only one can be expanded in hypertree
    - Backup in this case will not go back to the second parent, however proofs/status of nodes are propagated 
    automatically to all parents
    
- Expandable goals are either open or proven (as paper allows proven goals to be explored)