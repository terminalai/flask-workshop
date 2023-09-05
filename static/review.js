// The review div, which will be populated with the reviews
const reviewsDiv = document.getElementById("reviews_div");

// Fetch and display all the review information
async function populate_reviews() {
    let reviews = await (await fetch(`/reviews`)).json();
    reviews.forEach(review => {
        console.log(review)
        
        // Create a Div for the review element
        const reviewDisp = document.createElement('div');
        reviewDisp.classList.add("review");
        reviewDisp.innerHTML = `<p class="review-text">${review.review.replaceAll("\r", "<br>")}</p>`;

        // The stars
        reviewDisp.innerHTML += `<span class="review-star fa fa-star${review.score == 1 ? "-half-o" : review.score > 1 ? "" : "-o"}"></span>\n`;
        reviewDisp.innerHTML += `<span class="review-star fa fa-star${review.score == 3 ? "-half-o" : review.score > 3 ? "" : "-o"}"></span>\n`;
        reviewDisp.innerHTML += `<span class="review-star fa fa-star${review.score == 5 ? "-half-o" : review.score > 5 ? "" : "-o"}"></span>\n`;
        reviewDisp.innerHTML += `<span class="review-star fa fa-star${review.score == 7 ? "-half-o" : review.score > 7 ? "" : "-o"}"></span>\n`;
        reviewDisp.innerHTML += `<span class="review-star fa fa-star${review.score == 9 ? "-half-o" : review.score > 9 ? "" : "-o"}"></span>`;

        reviewsDiv.prepend(reviewDisp);
    });

}

populate_reviews();