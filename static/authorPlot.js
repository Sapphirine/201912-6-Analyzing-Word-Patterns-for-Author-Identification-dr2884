d3.selection.prototype.moveToFront = function() {
  return this.each(function(){
    this.parentNode.appendChild(this);
  });
};

function authorPlot(id, fData) {

    // Define the color to change if your mouse move on the bar
    var barColor = '#4753CC';

    distinctAuthors = [...new Set(fData.map(d => d.author))]

    // Choose color for each word:
    function authorColor(author) {
        cmap = {0: "#CC6E47", 1: "#73C9FF", 2: "#828499", 3: '#FFD4B3'};
        return cmap[distinctAuthors.indexOf(author)]
    }

    // function to handle histogram.
    function scatterPlot(fD) {
        var hG = {}, hGDim = {t: 60, r: 0, b: 30, l: 0};
        hGDim.w = 800 - hGDim.l - hGDim.r,
            hGDim.h = 600 - hGDim.t - hGDim.b;

        //create svg for histogram.
        var hGsvg = d3.select(id).append("svg")
            .attr("width", hGDim.w + hGDim.l + hGDim.r)
            .attr("height", hGDim.h + hGDim.t + hGDim.b)
            .attr("border", 1)
            .append("g")
            .attr("transform", "translate(" + hGDim.l + "," + hGDim.t + ")");

        var borderPath = hGsvg.append("rect")
       			.attr("height", 480)
       			.attr("width", 800)
                .attr("transform", "translate(" + hGDim.l + "," + hGDim.t + ")")
       			.style("stroke", "black")
       			.style("fill", "none")
       			.style("stroke-width", 2);


        var xData = fData.map(d => d.x)
        var yData = fData.map(d => d.y)

        var x = d3.scale.linear()
               .domain([d3.min(xData), d3.max(xData)])
               .range([0+100, hGDim.w-100])

        // Add x-axis to the histogram svg.
        hGsvg.append("g").attr("class", "x axis")
            .attr("transform", "translate(0," + hGDim.h + ")")
            .call(d3.svg.axis().scale(x).orient("bottom"));

        var y = d3.scale.linear()
               .domain([d3.min(yData), d3.max(yData)])
               .range([hGDim.h-100, 0+100])

        // Create bars for histogram to contain rectangles and count labels.
        var circles = hGsvg.selectAll(".circle").data(fD).enter()
            .append("g").attr("class", "circle");

        //create the rectangles.
        circles.append("circle")
            .attr("cx", function(d) {return x(d.x);})
            .attr("cy", function(d) {return y(d.y);})
            .attr("r", 6)
            .attr('fill', function(d) {return authorColor(d.author)})
            .attr('stroke', 'black')
            .attr('stroke-width', '1')
            // .on("mouseover", mouseover)
            // .on("mouseout", mouseout);

        //Create the frequency labels ABOVE the rectangles.
        circles.append("text")
            .text(function (d) {
            return d.title
        })
            .attr("x", function (d) {
                return x(d.x);
            })
            .attr("y", function(d){return y(d.y)-2})
            .attr("text-anchor", "middle")
            .attr('font-weight', 'bold')
            .attr('stroke', 'white')
            .attr('stroke-width', '0.5')
            .style("visibility", "hidden");

        circles.on("mouseover", function(d) {
                d3.select(this).select("text").style("visibility", "visible");
                d3.select(this).moveToFront();
            });

        circles.on("mouseout", function(d) {
                d3.select(this).select("text").style("visibility", "hidden");
            });
        return hG;
    }

    // function to handle legend.
    function legend(lD) {
        var leg = {};

        // create table for legend.
        var legend = d3.select(id).append("table").attr('stroke', 'black')
            .attr('stroke-width', '1').attr('class', 'legend');

        // create one row per segment.
        var tr = legend.append("tbody").selectAll("tr").data(lD).enter().append("tr");

        // create the first column for each segment.
        tr.append("td")
            .append("svg")
            .attr("width", '16')
            .attr("height", '16')
            .append("rect")
            .attr("width", '16').attr("height", '16')
            .attr("fill", function (d) {
                return authorColor(d.type);
            });

        // create the second column for each segment.
        tr.append("td").text(function (d) {
            return d.type;
        });

        // create the third column for each segment.
        tr.append("td").attr("class", 'legendFreq')
            .text(function (d) {
                return d3.format(",")(d.count);
            });

        // create the fourth column for each segment.
        tr.append("td").attr("class", 'legendPerc')
            .text(function (d) {
                return getLegend(d, lD);
            });

        // Utility function to be used to update the legend.
        leg.update = function (nD) {
            // update the data attached to the row elements.
            var l = legend.select("tbody").selectAll("tr").data(nD);

            // update the frequencies.
            l.select(".legendFreq").text(function (d) {
                return d3.format(",")(d.count);
            });

            // update the percentage column.
            l.select(".legendPerc").text(function (d) {
                return getLegend(d, nD);
            });
        }

        function getLegend(d, aD) { // Utility function to compute percentage.
            return d3.format("%")(d.count / d3.sum(aD.map(function (v) {
                return v.count;
            })));
        }

        return leg;
    }

    // calculate total count by segment for all state.

    var tF = distinctAuthors.map(function(a) {
        return {
            type: a, count: fData.filter(function (t) {
                return t.author == a;
            }).length
        };
    });


    var hG = scatterPlot(fData), // create the scatterPlot.
        leg = legend(tF); // create the legend.

}