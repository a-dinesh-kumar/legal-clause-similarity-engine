const queryInput = document.getElementById("query");
const topKInput = document.getElementById("topK");
const searchButton = document.getElementById("searchButton");

const status = document.getElementById("status");
const resultsList = document.getElementById("resultsList");


searchButton.addEventListener("click", searchClauses);


async function searchClauses() {

    const query = queryInput.value.trim();
    const topK = Number(topKInput.value);

    if (!query) {

        status.textContent = "Please enter a legal clause.";
        status.className = "status error";

        resultsList.innerHTML = "";

        return;
    }


    searchButton.disabled = true;

    status.textContent = "Searching for similar clauses...";
    status.className = "status";

    resultsList.innerHTML = "";


    try {

        const response = await fetch("/api/search", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                query: query,
                top_k: topK
            })

        });


        if (!response.ok) {

            const error = await response.json();

            throw new Error(
                error.detail || "Search failed."
            );
        }


        const data = await response.json();

        displayResults(data.results);

        status.textContent =
            `${data.results.length} similar clauses found.`;

        status.className = "status success";


    } catch (error) {

        status.textContent = error.message;
        status.className = "status error";

    } finally {

        searchButton.disabled = false;
    }
}


function displayResults(results) {

    resultsList.innerHTML = "";


    if (!results.length) {

        resultsList.innerHTML =
            '<p class="empty">No similar clauses found.</p>';

        return;
    }


    results.forEach(result => {

        const card = document.createElement("article");

        card.className = "result-card";


        // Convert cosine similarity from 0-1
        // into a human-readable percentage.
        const similarityPercentage =
            (result.similarity_score * 100).toFixed(2);


        card.innerHTML = `

            <div class="result-top">

                <span class="rank">
                    #${result.rank}
                </span>

                <span class="score">
                    Similarity: ${similarityPercentage}%
                </span>

            </div>


            <span class="clause-type">
                ${escapeHtml(result.clause_type)}
            </span>


            <p class="clause-text">
                ${escapeHtml(result.clause_text)}
            </p>

        `;


        resultsList.appendChild(card);
    });
}


function escapeHtml(value) {

    const div = document.createElement("div");

    div.textContent = value;

    return div.innerHTML;
}