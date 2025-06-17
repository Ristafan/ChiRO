import os

import pandas as pd
from typing import List, Optional, Tuple, Dict
import seaborn as sns
from matplotlib import pyplot as plt

from src.DataSetSplit.TrainingClasses import bat_species_fixed


class DatasetSplitter:
    def __init__(
            self,
            excel_path: str,
            root_path: str,
            col_filename: str = "File",
            col_location: str = "location",
            col_label: str = "Verification 1",
            split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
            seed: Optional[int] = None,
            class_sample_limit: Optional[int] = None,
            use_min_class_count: bool = False,
            balance_by_location: bool = False,
    ):
        self.excel_path = excel_path
        self.root_path = root_path
        self.col_filename = col_filename
        self.col_location = col_location
        self.col_label = col_label

        self.split_ratios = split_ratios
        self.seed = seed
        self.class_sample_limit = class_sample_limit
        self.use_min_class_count = use_min_class_count
        self.balance_by_location = balance_by_location

        self.df: Optional[pd.DataFrame] = None
        self.merged_labels: Dict[str, str] = {}  # maps old_label -> merged_label

        # After splitting
        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.class_counts_after_limit: Dict[str, int] = {}

    def load_data(self):
        self.df = pd.read_excel(self.excel_path)

    def merge_labels(self, merge_groups: List[List[str]]):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        for group in merge_groups:
            if not group:
                continue
            new_label = group[0].split()[0]
            for old_label in group:
                self.merged_labels[old_label] = new_label

        # Apply merging
        self.df['Class'] = self.df[self.col_label].apply(lambda lbl: self.merged_labels.get(lbl, lbl))

        # Assign integer labels to each class
        class_to_label = {cls_name: idx for idx, cls_name in enumerate(sorted(self.df['Class'].unique()))}
        self.df['label'] = self.df['Class'].map(class_to_label)

    def _min_class_count(self) -> int:
        counts = self.df['Class'].value_counts()
        return counts.min()

    def _get_sampled_class_data(self, class_name: str) -> pd.DataFrame:
        class_df = self.df[self.df['Class'] == class_name]
        available_count = len(class_df)

        if self.use_min_class_count:
            min_count = self._min_class_count()
            n_samples = min(min_count, available_count)
        elif self.class_sample_limit is not None:
            n_samples = min(self.class_sample_limit, available_count)
        else:
            n_samples = available_count

        if n_samples < available_count:
            sampled_df = class_df.sample(n=n_samples, random_state=self.seed).reset_index(drop=True)
        else:
            sampled_df = class_df.reset_index(drop=True)

        self.class_counts_after_limit[class_name] = n_samples
        return sampled_df

    def _split_class_data(self, class_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not self.balance_by_location:
            df_shuffled = class_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            n = len(df_shuffled)
            train_end = int(n * self.split_ratios[0])
            val_end = train_end + int(n * self.split_ratios[1])
            return df_shuffled.iloc[:train_end], df_shuffled.iloc[train_end:val_end], df_shuffled.iloc[val_end:]

        # balance by location: shuffle per location but split globally
        parts = []
        locations = class_df[self.col_location].unique()

        # Shuffle and collect all data, preserving location balance in shuffle
        shuffled_parts = []
        for loc in locations:
            loc_df = class_df[class_df[self.col_location] == loc]
            loc_df = loc_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            shuffled_parts.append(loc_df)
        all_shuffled_df = pd.concat(shuffled_parts).sample(frac=1, random_state=self.seed).reset_index(drop=True)

        n = len(all_shuffled_df)
        train_end = int(n * self.split_ratios[0])
        val_end = train_end + int(n * self.split_ratios[1])

        train_df = all_shuffled_df.iloc[:train_end]
        val_df = all_shuffled_df.iloc[train_end:val_end]
        test_df = all_shuffled_df.iloc[val_end:]

        return train_df, val_df, test_df

    def create_splits(self):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.class_counts_after_limit.clear()

        for class_name in self.df['Class'].unique():
            sampled_class_df = self._get_sampled_class_data(class_name)
            train_df, val_df, test_df = self._split_class_data(sampled_class_df)

            self.train_df = pd.concat([self.train_df, train_df], ignore_index=True)
            self.val_df = pd.concat([self.val_df, val_df], ignore_index=True)
            self.test_df = pd.concat([self.test_df, test_df], ignore_index=True)

        # Shuffle final splits
        self.train_df = self.train_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.val_df = self.val_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.test_df = self.test_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        max_count = max(self.class_counts_after_limit.values(), default=0)
        if any(count < max_count for count in self.class_counts_after_limit.values()):
            print("\n⚠️ Warning: Dataset is imbalanced due to sampling limits:")
            self.print_imbalance_report()

        # Return number of classes
        return len(self.df['Class'].unique())

    def print_imbalance_report(self):
        for class_name, count in self.class_counts_after_limit.items():
            print(f"Class '{class_name}': {count} samples")

    def export_splits_to_excel(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        def prepare_export_df(df):
            if self.col_filename in df.columns:
                df = df.copy()
                df['Filename'] = df[self.col_filename]
            else:
                df['Filename'] = 'unknown'

            def build_filepath(row):
                original_path = row.get('output_fpath', 'unknown')
                if original_path == 'unknown' or not isinstance(original_path, str):
                    return 'unknown'

                # Normalize root path
                new_root = self.root_path.replace('\\', '/')
                if not new_root.endswith('/'):
                    new_root += '/'

                # Extract the relative path from the original output_fpath
                # (everything after "LabelledSequences")
                split_token = "LabelledSequences"
                if split_token in original_path:
                    _, relative_path = original_path.split(split_token, 1)
                    relative_path = relative_path.lstrip("\\/")
                    # Construct new path
                    return f"{new_root}{relative_path}".replace('\\', '/')

                return 'unknown'

            df['Filepath'] = df.apply(build_filepath, axis=1)

            # Include required columns (modify as needed)
            export_df = df[[
                'Filename',
                'Filepath',
                self.col_location,
                self.col_label,
                'Class',
                'label'
            ]].copy()

            return export_df

            # Consistent labels 0 to n-1 after merging/ignoring
            class_list = sorted(self.df['Class'].unique())  # All final classes
            label_map = {cls_name: idx for idx, cls_name in enumerate(class_list)}
            df['label'] = df['Class'].map(label_map)

        sets = {
            'train': self.train_df,
            'val': self.val_df,
            'test': self.test_df
        }

        for set_name, df in sets.items():
            if df is None or df.empty:
                print(f"Warning: {set_name} set is empty or not available, skipping export.")
                continue

            df_prepared = prepare_export_df(df)

            # Now build export df with all needed columns, pulling from df_prepared:
            export_df = pd.DataFrame({
                'Filename': df_prepared['Filename'],
                'Filepath': df_prepared['Filepath'],
                'location': df_prepared.get(self.col_location, pd.Series(['unknown'] * len(df_prepared))),
                'Verification 1': df_prepared.get(self.col_label, pd.Series(['unknown'] * len(df_prepared))),
                'Class': df_prepared.get('Class', df_prepared.get(self.col_label, pd.Series(['unknown'] * len(df_prepared)))),
                'label': df_prepared.get('label', pd.Series([0] * len(df_prepared)))
            })

            filepath = os.path.join(output_dir, f"{set_name}_dataset_info.xlsx")
            export_df.to_excel(filepath, index=False)
            print(f"Exported {set_name} set to {filepath}")

    def plot_split_distribution_stacked_bar_chart(self, show_plots: bool = True):
        if not show_plots:
            return

        if self.train_df.empty or self.val_df.empty or self.test_df.empty:
            print("Split dataframes are empty. Please run create_splits() first.")
            return

        # Helper to get stacked bar data
        def get_counts(df, group_col):
            counts = df.groupby(group_col).size()
            return counts

        sets = ['Train', 'Validation', 'Test']
        dfs = [self.train_df, self.val_df, self.test_df]

        # Plot 1: Verification 1 classes stacked bar chart
        plt.figure(figsize=(10, 10))
        counts_list = [get_counts(df, self.col_label) for df in dfs]
        all_labels = sorted(set().union(*[c.index for c in counts_list]))

        bottoms = [0]*len(sets)
        for label in all_labels:
            heights = [counts.get(label, 0) for counts in counts_list]
            plt.bar(sets, heights, bottom=bottoms, label=label)
            bottoms = [sum(x) for x in zip(bottoms, heights)]

        plt.title('Data Split Distribution by Verification 1 Class (stacked)')
        plt.ylabel('Number of files')
        plt.xlabel('Dataset Split')
        plt.legend(title='Verification 1', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        # Plot 2: Location stacked bar chart (only if balancing by location)
        if self.balance_by_location:
            plt.figure(figsize=(10, 6))
            counts_list_loc = [get_counts(df, self.col_location) for df in dfs]
            all_locations = sorted(set().union(*[c.index for c in counts_list_loc]))

            bottoms = [0]*len(sets)
            for location in all_locations:
                heights = [counts.get(location, 0) for counts in counts_list_loc]
                plt.bar(sets, heights, bottom=bottoms, label=location)
                bottoms = [sum(x) for x in zip(bottoms, heights)]

            plt.title('Data Split Distribution by Location (stacked)')
            plt.ylabel('Number of files')
            plt.xlabel('Dataset Split')
            plt.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()

    def plot_split_distribution(self, show_plots: bool = True):
        if not show_plots:
            return

        if self.train_df.empty or self.val_df.empty or self.test_df.empty:
            print("Split dataframes are empty. Please run create_splits() first.")
            return

        # Prepare data for plotting
        def prepare_plot_data():
            dfs = [self.train_df, self.val_df, self.test_df]
            sets = ['Train', 'Validation', 'Test']
            combined = []

            for set_name, df in zip(sets, dfs):
                counts_label = df[self.col_label].value_counts().reset_index()
                counts_label.columns = ['Verification 1', 'Count']
                counts_label['Set'] = set_name

                counts_location = df[self.col_location].value_counts().reset_index()
                counts_location.columns = ['location', 'Count']
                counts_location['Set'] = set_name

                combined.append((counts_label, counts_location))

            return combined

        combined = prepare_plot_data()

        # Plot 1: Verification 1 class distribution across sets
        plt.figure(figsize=(12, 6))
        all_counts_label = pd.concat([c[0] for c in combined], ignore_index=True)
        sns.barplot(data=all_counts_label, x='Set', y='Count', hue='Verification 1', dodge=True)
        plt.title('Data Split Distribution by Verification 1 Class')
        plt.ylabel('Number of files')
        plt.xlabel('Dataset Split')
        plt.legend(title='Verification 1', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        # Plot 2: Location distribution across sets (only if balancing by location is enabled)
        if self.balance_by_location:
            plt.figure(figsize=(12, 6))
            all_counts_location = pd.concat([c[1] for c in combined], ignore_index=True)
            sns.barplot(data=all_counts_location, x='Set', y='Count', hue='location', dodge=True)
            plt.title('Data Split Distribution by Location')
            plt.ylabel('Number of files')
            plt.xlabel('Dataset Split')
            plt.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # Set up your paths and merge groups here
    excel_path = "D:/Bachelorarbeit/AgroscopeData/LabelledSequencesMerged_cleaned_cleaned_cleaned.xlsx"
    root_path = "D:/Bachelorarbeit/AgroscopeData/LabelledSequences"

    splitter = DatasetSplitter(
        excel_path=excel_path,
        root_path=root_path,
        seed=42,
        class_sample_limit=50,
        use_min_class_count=False,
        balance_by_location=True
    )

    splitter.load_data()
    splitter.merge_labels([bat_species_fixed])
    print(splitter.df['Class'].value_counts())
    print("Class distribution after merging:")
    print(splitter.df['Class'].value_counts())

    splitter.create_splits()
    print("Class counts after sampling:")
    print(splitter.class_counts_after_limit)

    if False:
        splitter.plot_split_distribution_stacked_bar_chart(show_plots=True)
        splitter.plot_split_distribution(show_plots=True)

    splitter.export_splits_to_excel("D:/Bachelorarbeit/AgroscopeData")

    print(f"Train samples: {len(splitter.train_df)}")
    print(f"Validation samples: {len(splitter.val_df)}")
    print(f"Test samples: {len(splitter.test_df)}")
