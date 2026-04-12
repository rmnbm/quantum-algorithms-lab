from __future__ import annotations

"""Branch-and-bound baseline for 0/1 knapsack instances."""


class BranchAndBound:
    """Solve 0/1 knapsack instances with branch and bound."""

    def __init__(self, items, values, weights):
        if len(items) != len(values) or len(values) != len(weights):
            raise ValueError("items, values, and weights must have the same length.")

        self.items = list(items)
        self.values = list(values)
        self.weights = list(weights)
        self.item_count = len(values)
        self.sorted_items = self._sort_items()

    def _sort_items(self):
        ranked = []
        for index, (item, value, weight) in enumerate(zip(self.items, self.values, self.weights)):
            ratio = float("inf") if weight == 0 else value / weight
            ranked.append(
                {
                    "index": index,
                    "item": item,
                    "weight": weight,
                    "value": value,
                    "ratio": ratio,
                }
            )
        return sorted(ranked, key=lambda entry: entry["ratio"], reverse=True)

    def calculate_upper_bound(self, index, remaining_capacity, current_value):
        bound = current_value
        available = remaining_capacity
        cursor = index

        while cursor < self.item_count and self.sorted_items[cursor]["weight"] <= available:
            available -= self.sorted_items[cursor]["weight"]
            bound += self.sorted_items[cursor]["value"]
            cursor += 1

        if cursor < self.item_count:
            bound += available * self.sorted_items[cursor]["ratio"]

        return bound

    def _explore(self, index, remaining_capacity, current_value, current_selection):
        if index == self.item_count:
            if current_value > self.best_value:
                self.best_value = current_value
                self.best_selection = list(current_selection)
            return

        if self.calculate_upper_bound(index, remaining_capacity, current_value) <= self.best_value:
            return

        item = self.sorted_items[index]
        if item["weight"] <= remaining_capacity:
            current_selection[index] = 1
            self._explore(
                index + 1,
                remaining_capacity - item["weight"],
                current_value + item["value"],
                current_selection,
            )

        current_selection[index] = 0
        self._explore(index + 1, remaining_capacity, current_value, current_selection)

    def solve(self, capacity):
        self.best_value = 0
        self.best_selection = [0] * self.item_count
        self._explore(0, capacity, 0, [0] * self.item_count)

        original_order_selection = [0] * self.item_count
        for sorted_index, bit in enumerate(self.best_selection):
            if bit:
                original_index = self.sorted_items[sorted_index]["index"]
                original_order_selection[original_index] = 1

        return original_order_selection, self.best_value

    # Legacy compatibility wrappers kept for the original course notebooks.
    def resoudre(self, poids_max):
        return self.solve(poids_max)

    def trier_items(self):
        return self._sort_items()

    def calculer_borne_sup(self, index, poids_restant, valeur_actuelle):
        return self.calculate_upper_bound(index, poids_restant, valeur_actuelle)

    def explorer(self, index, poids_restant, valeur_actuelle, selection_actuelle):
        return self._explore(index, poids_restant, valeur_actuelle, selection_actuelle)
