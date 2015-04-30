from ml.classifier.decision_tree import BinaryTree, CategoricalTree, EntropyMixin

__author__ = 'jamesmcnamara'


class CategoricalEntropyTree(CategoricalTree, EntropyMixin):
    def classify(self, observation):
        """
            Consumes an observation and outputs the classification that this
            tree would apply to the row
        :param observation: a row of observational data
        :return: the label that would be applied to the given row
        """
        if self.split_on.column != -1:
            return self.children[observation[self.split_on.column]].classify(observation)
        else:
            return self.classification

    def __str__(self):
        return super().__str__()


class EntropyTree(BinaryTree, EntropyMixin):
    def classify(self, row):
        """
            Consumes an observation returns the classification of that
            observation via this tree and if this tree is a leaf, determines 1
            if this observation was classified correctly else 0 else uses this
            trees splitter to determine which sub-tree to delegate to, and
            then returns whether the subtree correctly classified the node
        :param row_and_result: a 2-tuple of observational data (1D array) and
            the result for that observation
        :return: 1 if this tree correctly classified the input else 0
        """
        if self.split_on.splitter:
            if self.split_on.splitter(row[self.split_on.column]):
                return self.left.classify(row)
            else:
                return self.right.classify(row)
        else:
            return self.classification

    def __str__(self):
        """
            OVERRIDE: str(self) returns a pretty printed version of the tree
        """
        offset = "\n" + "\t" * self.depth()
        s = "{off}Total Entries: {len},{off}Entropy: {ent:.3f}".format(
            off=offset, len=len(self),
            ent=self.measure_function(self.results),
            col=self.split_on.column)
        if self.split_on_column != 1:
            s += ",{off}Split on column: {col}".format(
                off=offset, col=self.split_on.column)
        else:
            s += ""

        for branch, branch_name in [(self.left, "Left"), (self.right, "Right")]:
            if branch: 
                s += "{off}__________________________{off}{name}:{branch}".format(off=offset, name=branch_name, branch=str(branch))
        return s


