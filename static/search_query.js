
// function to send a search request to the server
async function search_query() {
    let search_query_input = document.getElementById("search_query").value;
    const response = await fetch(url + '/search_query/${search_query_input}')

    const results_div = document.getElementById("results");
    results_dic.inner = response.text()
}