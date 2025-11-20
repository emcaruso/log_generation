import os
import json
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import joblib

src_dir = Path(os.path.dirname(os.path.abspath(__file__)))
tree_data_dir = src_dir / ".." / "data" / "tree_geo_original"
stats_filepath = (
    src_dir / ".." / "data" / "tree_statistics" / "tree_data_statistics.json"
)


def main():
    # collect data
    data = {}
    for filename in tqdm(os.listdir(str(tree_data_dir)), desc="Loading tree data"):
        if filename.endswith(".json") and not filename.endswith("params.json"):

            filepath = tree_data_dir / filename
            name = filepath.stem
            with open(filepath, "r") as f:
                data[name] = np.array(json.load(f))

    # statistics
    stats = get_stats(data)

    # gmms for knots
    plot_knots_data(data["geo_knots"])
    result_knots = get_knots_gmms(data["geo_knots"])
    plot_gmm_distributions(data["geo_knots"], result_knots)
    save_gmm(result_knots)

    # save statistics
    with open(stats_filepath, "w") as f:
        json.dump(stats, f, indent=4)


def plot_gmm_distributions(data, results):
    dir = src_dir / ".." / "data" / "tree_statistics"
    os.makedirs(dir / "gmm_distributions", exist_ok=True)
    n_dims = data.shape[1]

    for i in range(n_dims):
        dim_data = data[:, i].reshape(-1, 1)
        gmm_results = results[f"dim_{i+1}"]

        x = np.linspace(np.min(dim_data), np.max(dim_data), 1000).reshape(-1, 1)

        plt.figure(figsize=(5, 3))
        plt.hist(dim_data, bins=30, density=True, alpha=0.5, color="gray")

        best_model_aic = gmm_results["best_model_aic"]
        logprob_aic = best_model_aic.score_samples(x)
        pdf_aic = np.exp(logprob_aic)
        plt.plot(x, pdf_aic, "-r", label=f"GMM AIC (k={gmm_results['best_k_aic']})")

        best_model_bic = gmm_results["best_model_bic"]
        logprob_bic = best_model_bic.score_samples(x)
        pdf_bic = np.exp(logprob_bic)
        plt.plot(x, pdf_bic, "-b", label=f"GMM BIC (k={gmm_results['best_k_bic']})")

        single_gauss_model = gmm_results["models"][0]
        logprob_single = single_gauss_model.score_samples(x)
        pdf_single = np.exp(logprob_single)
        plt.plot(x, pdf_single, "-g", label="Single Gaussian")

        plt.title(f"knots - Dimension {i+1}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            dir / "gmm_distributions" / f"knots_dim{i+1}_gmm.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


def choose_best_gmm(X, max_components=5, covariance_type="full"):
    models = []
    aics = []
    bics = []

    for k in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type=covariance_type, random_state=42
        )
        gmm.fit(X)

        models.append(gmm)
        aics.append(gmm.aic(X))
        bics.append(gmm.bic(X))

    best_k_aic = np.argmin(aics) + 1
    best_k_bic = np.argmin(bics) + 1

    return {
        "models": models,
        "aic": aics,
        "bic": bics,
        "best_k_aic": best_k_aic,
        "best_k_bic": best_k_bic,
        "best_model_aic": models[np.argmin(aics)],
        "best_model_bic": models[np.argmin(bics)],
    }


def get_stats(data):
    stats = {}
    for name in tqdm(data.keys(), desc="Computing statistics"):
        stats[name] = {}
        d = data[name]

        if name == "geo_radii":
            d = d.flatten()
        stats[name]["mean"] = d.mean(axis=0).tolist()
        stats[name]["std"] = d.std(axis=0).tolist()
        stats[name]["min"] = d.min(axis=0).tolist()
        stats[name]["max"] = d.max(axis=0).tolist()
    return stats


def get_knots_gmms(data):
    n_dims = data.shape[1]
    stats = {}
    results = {}
    for i in range(n_dims):
        dim_data = data[:, i].reshape(-1, 1)
        gmm_results = choose_best_gmm(
            dim_data, max_components=5, covariance_type="full"
        )
        stats[f"dim_{i+1}"] = {
            "best_k_aic": gmm_results["best_k_aic"],
            "best_k_bic": gmm_results["best_k_bic"],
        }
        results[f"dim_{i+1}"] = gmm_results
    return results


def plot_knots_data(data):
    # plot histograms
    FIG_WIDTH = 5  # wider images
    FIG_HEIGHT_PER_ROW = 1
    TITLE_FONTSIZE = 5
    LABEL_FONTSIZE = 4
    TICK_FONTSIZE = 3

    n_dims = data.shape[1]

    # Make the figure scale with number of rows
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT_PER_ROW * n_dims))

    for i in tqdm(range(n_dims), desc=f"Plotting histograms for knots"):

        ax = plt.subplot(n_dims, 1, i + 1)

        # Histogram
        ax.hist(data[:, i], bins=30, alpha=0.7, color="blue")

        # Smaller text
        ax.set_title(f"knots - Dimension {i+1}", fontsize=TITLE_FONTSIZE)
        ax.set_xlabel("Value", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Frequency", fontsize=LABEL_FONTSIZE)

        # Tick size
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    plt.tight_layout()

    plt.savefig(
        src_dir / ".." / "data" / "tree_statistics" / f"knots_histograms.png",
        dpi=150,  # ADDS HIGH RESOLUTION (recommended!)
        bbox_inches="tight",
    )
    plt.close()


# save gmm with joblib
def save_gmm(results):
    dir = src_dir / ".." / "data" / "tree_statistics" / "gmm_models"
    os.makedirs(dir, exist_ok=True)
    n_dims = len(results.keys())
    for i in tqdm(range(n_dims), "Saving GMM models for knots"):
        best_model_aic = results[f"dim_{i+1}"]["best_model_aic"]
        best_model_bic = results[f"dim_{i+1}"]["best_model_bic"]
        single_gauss_model = results[f"dim_{i+1}"]["models"][0]
        joblib.dump(
            best_model_aic,
            dir / f"aic_dim{i+1}.pkl",
        )
        joblib.dump(
            best_model_bic,
            dir / f"bic_dim{i+1}.pkl",
        )
        joblib.dump(
            single_gauss_model,
            dir / f"single_dim{i+1}.pkl",
        )


if __name__ == "__main__":
    main()
