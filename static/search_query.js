const url = window.location.origin + "/"


// building the dropdown menus
async function options_innit() {
    const response = await fetch(url + 'get_dropdown_options')
    const dropdown_options_dic = await response.json()

    const monomerSel = document.getElementById("monomer");
    const raft_agentSel = document.getElementById("raft_agent");
    const solventSel = document.getElementById("solvent");

    const dropdownSelectors = {"monomer": monomerSel, "raft_agent": raft_agentSel, "solvent": solventSel}
    const keys = ["monomer", "raft_agent", "solvent"]

    for (const dropdown of keys){
        const dropdownSel = dropdownSelectors[dropdown]
        for (const option of dropdown_options_dic[dropdown]){
            dropdownSel.options[dropdownSel.options.length] = new Option(option, option);
    }
    }
}

window.onload = function() {
  options_innit().then(r => console.log("Dropdowns initialized"))
}


// function to send a plot request to the server
async function plot_query() {
    let plot_query_input = document.getElementById("search").value;
    const response = await fetch(url + '/plot_query/${plot_query_input}')

    const results_div = document.getElementById("results");
    results_div.inner = response.text()

}
