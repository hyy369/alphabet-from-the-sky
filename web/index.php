<!doctype html>
<html lang="en">
<head>
  <title>Alphabet from the Sky</title>
  <meta name="Author" content="Yangyang He">
  <meta content="width=device-width,initial-scale=1" name=viewport>
  <meta content="text/html;charset=utf-8" http-equiv="Content-Type">
  <!-- Latest compiled and minified CSS -->
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css" integrity="sha384-BVYiiSIFeK1dGmJRAkycuHAHRg32OmUcww7on3RYdg4Va+PmSTsz/K68vbdEjh4u" crossorigin="anonymous">
  <link rel="stylesheet" href="assets/css/main.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.0/jquery.min.js"></script>
  <script src="https://netdna.bootstrapcdn.com/bootstrap/3.0.0/js/bootstrap.min.js"></script>
  <!-- <script src="assets/javascript/md5.js"></script> -->
  <!-- <script src="assets/javascript/search.js"></script> -->
</head>
<body>

  <div class="jumbotron jumbotron-fluid center">
    <div class="container">

      <!-- <img id="icon-marvel-logo-svg" src="assets/img/marvel.svg"/> -->
      <h1 class="display-3">Alphabet from the Sky</h1>

      <div class="search">
        <!-- <input type="text" class="search_box" id="title_box" placeholder="Type to start..."> -->
        <form action="index.php" method="post">
            <select class="dropdown" name="character" id="character">
                <option value="A">A</option>
                <option value="B">B</option>
                <option value="C">C</option>
                <option value="D">D</option>
                <option value="E">E</option>
                <option value="F">F</option>
                <option value="G">G</option>
                <option value="H">H</option>
                <option value="I">I</option>
                <option value="J">J</option>
                <option value="K">K</option>
                <option value="L">L</option>
                <option value="M">M</option>
                <option value="N">N</option>
                <option value="O">O</option>
                <option value="P">P</option>
                <option value="Q">Q</option>
                <option value="R">R</option>
                <option value="S">S</option>
                <option value="T">T</option>
                <option value="U">U</option>
                <option value="V">V</option>
                <option value="W">W</option>
                <option value="X">X</option>
                <option value="Y">Y</option>
                <option value="Z">Z</option>
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
                <option value="6">6</option>
                <option value="7">7</option>
                <option value="8">8</option>
                <option value="9">9</option>
                <option value="!">!</option>
                <option value="?">?</option>
                <option value=":">:</option>
                <option value=".">.</option>
            </select>
            <script type="text/javascript">
                document.getElementById('character').value = "<?php if ($_POST['character']) echo $_POST['character']; else echo 'A';?>";
            </script>

            <select class="dropdown" name="f_color" id="f_color">
                <option value="0">Red</option>
                <option value="30">Red-Yellow</option>
                <option value="60">Yellow</option>
                <option value="90">Yellow-Green</option>
                <option value="120">Green</option>
                <option value="180">Green-Blue</option>
                <option value="240">Blue</option>
                <option value="270">Blue-Purple</option>
                <option value="300">Purple</option>
                <option value="330">Purple-Red</option>
                <option value="BW">White-Gray-Black</option>
            </select>
            <script type="text/javascript">
                document.getElementById('f_color').value = "<?php if ($_POST['f_color']) echo $_POST['f_color']; else echo '120';?>";
            </script>

            <select class="dropdown" name="label" id="label">
            <option value="0">Urban</option>
            <option value="1">Rural</option>
            </select>
            <script type="text/javascript">
                document.getElementById('label').value = "<?php if ($_POST['label']) echo $_POST['label']; else echo '0';?>";
            </script>

            <br>
            <br>
            <button class="button" id="query_button" type="submit">Search</button>
        </form>

      </div>

      <p class="attribution">Created by <a href="https://hyy369.com/">Yangyang He</a> | <a href="">References</a> | Photo by NASA on Unsplash</p>
    </div>
  </div>

  <div class="container" id="result_header">

  <?php
    if ($_POST['character'])
    {
      // Connecting, selecting database
      $dbconn = pg_connect("host=localhost port=5433 dbname=hyy user=hyy password=")
        or die('Could not connect: ' . pg_last_error());

      // Performing SQL query
      if ($_POST["f_color"] == "BW")
        $query = "SELECT * FROM img2 WHERE character = '".$_POST["character"]."' AND label = ".$_POST["label"]." AND f_sat<=20 ORDER BY f_val DESC";
      else
        $query = "SELECT * FROM img2 WHERE character = '".$_POST["character"]."' AND label = ".$_POST["label"]." AND f_sat>20 ORDER BY 180-ABS(ABS(".$_POST["f_color"]."-f_hue)-180);";

      $result = pg_query($query) or die('Query failed: ' . pg_last_error());
      // Printing results in HTML
      while ($line = pg_fetch_array($result, null, PGSQL_ASSOC)) {
        echo '<div class="row">';
        echo '<div class="col-md-3">';
        echo '<img class="thumbnail" src="'.$line['file_path'].'"/>';
        echo '</div>';
        echo '<div class="col-md-3">';
        echo '<h4>Detected Central Color</h4>';
        echo '<div style="width:200px; height:120px; background-color: rgb('.$line[f_r].','.$line[f_g].','.$line[f_b].')"></div>';
        echo '</div>';
        echo '<div class="col-md-3">';
        echo '<h4>Detected Peripheral Color</h4>';
        echo '<div style="width:200px; height:120px; background-color: rgb('.$line[p_r].','.$line[p_g].','.$line[p_b].')"></div>';
        echo '</div>';
        echo '<div class="col-md-3">';
        echo '<h4>Predicted Environment</h4>';
        if ($line['label'] == 0)
          $label = "Urban";
        else
          $label = "Rural";
        echo '<h3>'.$label.'</h3>';
        echo '</div>';
        echo '</div>';
      }

      // Free resultset
      pg_free_result($result);

      // Closing connection
      pg_close($dbconn);
    }
  ?>

  </div>

  <div class="container" id="result">
  </div>

  
</body>
</html>
