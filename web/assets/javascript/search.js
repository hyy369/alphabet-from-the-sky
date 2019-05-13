var PRIV_KEY = "0a40841836e0697f38a8f3e8362cd42c5184a9ec";
var PUBLIC_KEY = "446bee50b1db8ed995dd97a138395991";
var COUNT = 0;

function changeSearchType(){
  $('#title_box').toggle();
  $('#initial_box').toggle();
}

function search()
{
  // Formulate API query
  var ts = new Date().getTime();
  var hash = md5(ts+PRIV_KEY + PUBLIC_KEY).toString();
  var comics_url = 'http://gateway.marvel.com:80/v1/public/comics';
  var char_url = 'http://gateway.marvel.com:80/v1/public/characters';

  var title = $('#title_box').val();
  var initial = $('#initial_box').val();
  var start_year = $('select[name="start_year"]').val();
  var format = $('select[name="format"]').val();
  var order_by = $('select[name="order_by"]').val();
  var query = {
      ts: ts,
      apikey: PUBLIC_KEY,
      hash: hash,
      limit: 100,
    };
  if ($('#title_radio').is(':checked'))
  {
    if (!title) {alert('Title cannot be empty!'); return;}
    query.title = title;
  }
  if ($('#initial_radio').is(':checked')){
    if (!initial) {alert('Initial letter cannot be empty!'); return;}
    query.titleStartsWith = initial;
  }
  if (start_year) {query.startYear = start_year;}
  if (format) {query.format = format;}
  if (order_by) {query.orderBy = order_by;}

  // Scroll to result div and display search status
  $('#result').empty();
  $('#result_header').empty().append('<div class="row center"><h1>Searching...</h1></div>');
  $([document.documentElement, document.body]).animate({scrollTop: $("#result_header").offset().top}, 2000);

  // Establish API call
  $.getJSON(comics_url, query)
  .done(function(data)
  {
    $('#result_header').empty().append('<div class="row center"><h1>Search results</h1></div>');
    for (var i = 0; i < data.data.count; i++)
    {
      if (data.data.results[i].characters.returned > 0)
      {
        var sorted_result = data.data.results[i].characters.items.sort(sort_name);
        getChar(sorted_result[0].name, data.data.results[i].title);
      }
      COUNT++;
    }
    if (COUNT === 0)
    {
      $('#result_header').empty();
      $('#result_header').append('<div class="row center"><h2>Sorry. No matching result was found.</h2></div>');
    }
  })
  .fail(function(err)
  {
    console.log(err);
    alert("Server side error.");
  })
}

function getChar(name, title)
{
  var ts = new Date().getTime();
  var hash = md5(ts+PRIV_KEY + PUBLIC_KEY).toString();
  var char_url = 'http://gateway.marvel.com:80/v1/public/characters';
  $.getJSON(char_url, {
    ts: ts,
    apikey: PUBLIC_KEY,
    hash: hash,
    limit: 100,
    name: name,
  })
  .done(function(data)
  {
    var description = data.data.results[0].description ? data.data.results[0].description : 'No description available.';
    $('#result').append
    (
      '<div class="row result-row"><div class="col-md-4">'+
      '<img class="thumbnail" src="'+data.data.results[0].thumbnail.path +'.'+ data.data.results[0].thumbnail.extension+'"/></div>'+
      '<div class="col-md-8"><h3>'+data.data.results[0].name+'</h3>'+
      '<h6>in <em>'+title+'</em></h6>'+
      '<p class="description">'+description+
      '</p></div></div>\n'
    );
    slicePage();
    $('#page_control').show();
    showControl();
  })
  .fail(function(err){
    console.log(err);
    alert("Server side error.");
  })
}

function sort_name (a, b)
{
  var x = a['name']; var y = b['name'];
  return ((x < y) ? -1 : ((x > y) ? 1 : 0));
}
