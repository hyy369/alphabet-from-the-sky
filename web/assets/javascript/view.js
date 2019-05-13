var $el = $('#result > div');
var pageSize = 10;
// slicePage();

function toggleMore()
{
  $('#more_options').show();
  $('#toggle_options').hide();
}

function slicePage()
{
  $el = $('#result > div');
  if ($el.length > pageSize)
  {
    $el.slice(0, pageSize).css({display: 'block'});
    $el.slice(pageSize, $el.length).css({display: 'none'});
  }
}

function addSlice(num)
{
  return num + pageSize;
}

function subtractSlice(num)
{
  return num - pageSize;
}

var slice = [0, pageSize];

function next()
{
  if (slice[1] < $el.length ){
    slice = slice.map(addSlice);
  }
  showControl();
  showSlice(slice);
}

function prev()
{
  if (slice[0] > 0 ){
    slice = slice.map(subtractSlice);
  }
  showControl();
  showSlice(slice);
}

function showControl() {
  $('#next').hide();
  $('#prev').hide();
  $('#sep').hide();
  if (slice[0] > 0) {$('#prev').show();}
  if (slice[1] < $el.length ) {$('#next').show();}
  if ($('#prev').is(":visible") && $('#next').is(":visible")) {$('#sep').show();}
}

function showSlice(slice)
{
  $([document.documentElement, document.body]).animate({scrollTop: $("#result_header").offset().top}, 1000);
  $el.css('display', 'none');
  $el.slice(slice[0], slice[1]).css('display','block');
}
