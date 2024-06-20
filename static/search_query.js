const url = window.location.origin + "/"

// building the dropdown menus
async function options_innit() {
    const response = await fetch(url + 'get_dropdown_options')
    const dropdown_options_dic = await response.json()

    const monomerSel = document.getElementById("monomer");
    const RAFT_agentSel = document.getElementById("RAFT-agent");
    const solventSel = document.getElementById("solvent");

    const dropdownSelectors = {"monomer": monomerSel, "RAFT-agent": RAFT_agentSel, "solvent": solventSel}
    const keys = ["monomer", "RAFT-agent", "solvent"]

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
async function search_query() {
    let search_query_input = document.getElementById("search_query").value;
    const response = await fetch(url + '/search_query/${search_query_input}')

    const results_div = document.getElementById("results");
    results_dic.inner = response.text()
}