import React, { useMemo } from 'react';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    TooltipProps
} from 'recharts';
import Icon from '@mdi/react';
import { mdiAlertCircleOutline } from '@mdi/js';
import { CombinedDataPoint, LoadedData, SenseInfo, PredictedVadPoint } from '../types';
import { getVADDescription, VADDimension } from '../vadUtils';

import '../styles/Plot2DVAD.scss';

interface Plot2DVADProps {
    selectedWords: string[];
    allWordsData: LoadedData | null;
    selectedSenseData?: SenseInfo | null;
    predictedVadSeries?: PredictedVadPoint[] | null;
}

interface PayloadEntry {
    color: string;
    dataKey: string;
    name: string;
    value: number | null;
    payload: CombinedDataPoint;
    stroke: string;
    fill: string;
    strokeDasharray?: string | number;
}

interface CustomTooltipProps extends TooltipProps<number | null, string> {
    active?: boolean;
    payload?: PayloadEntry[];
    label?: string | number;
}

interface ExtendedLineInfo {
    word: string;
    dim: VADDimension;
    color: string;
    dataKey: string;
    name: string;
    isForecast?: boolean;
}

interface MemoizedPlotData {
    chartData: CombinedDataPoint[];
    yDomain: [number | string, number | string];
    linesToPlot: ExtendedLineInfo[];
    hasValidData: boolean;
    timeDataRef: (number | string)[];
    showProportionLine: boolean;
}

const lineColors = ['#0d6efd', '#dc3545', '#198754', '#ffc107', '#6f42c1', '#fd7e14', '#20c997', '#6610f2'];
const forecastLineStyle = { strokeDasharray: "8 4", strokeWidth: 2 };

const Plot2DVAD: React.FC<Plot2DVADProps> = ({ selectedWords, allWordsData, selectedSenseData, predictedVadSeries }) => {

    const { chartData, yDomain, linesToPlot, hasValidData, timeDataRef, showProportionLine }: MemoizedPlotData = useMemo(() => {
        const workingChartDataMap = new Map<number, CombinedDataPoint>();
        const allValues: number[] = [];
        const lines: ExtendedLineInfo[] = [];
        let commonTimeRef: number[] | null = null;
        let validDataFound = false;
        let propData: (number | null)[] | null = null;
        let showProp = false;

        if (!allWordsData || selectedWords.length === 0) {
            return { chartData: [], yDomain: [0, 1] as [number | string, number | string], linesToPlot: [], hasValidData: false, timeDataRef: ['auto'], showProportionLine: false };
        }

        const singleWord = selectedWords.length === 1 ? selectedWords[0] : null;

        if (singleWord && selectedSenseData?.y_fitting) {
            const wordDataCheck = allWordsData ? allWordsData[singleWord] : null;
            if (wordDataCheck?.temporal_vad?.x && selectedSenseData.y_fitting.length === wordDataCheck.temporal_vad.x.length) {
                propData = selectedSenseData.y_fitting.map(p => (typeof p === 'number' && !isNaN(p) ? p : null));
                if (propData.some(p => p !== null)) {
                    showProp = true;
                }
            }
        }

        selectedWords.forEach((word, wordIndex) => {
            const wordData = allWordsData[word];
            if (!wordData?.temporal_vad?.x || !wordData?.temporal_vad?.v || !wordData?.temporal_vad?.a || !wordData?.temporal_vad?.d) {
                return;
            }

            const { x, v, a, d } = wordData.temporal_vad;

            if (!commonTimeRef) {
                commonTimeRef = [...x];
            } else {
                const currentWordTimesMatch = x.length === commonTimeRef.length && x.every((val, idx) => val === (commonTimeRef as number[])[idx]);
                if (!currentWordTimesMatch && !predictedVadSeries) {
                    console.warn(`Time data mismatch for word: ${word} and no forecast to extend. Skipping.`);
                    return;
                }
            }

            validDataFound = true;
            const colorBaseIndex = wordIndex % lineColors.length;
            const vColor = lineColors[(colorBaseIndex * 3) % lineColors.length];
            const aColor = lineColors[(colorBaseIndex * 3 + 1) % lineColors.length];
            const dColor = lineColors[(colorBaseIndex * 3 + 2) % lineColors.length];

            lines.push({ word, dim: 'Valence', color: vColor, dataKey: `${word}_V`, name: `${word} V`, isForecast: false });
            lines.push({ word, dim: 'Arousal', color: aColor, dataKey: `${word}_A`, name: `${word} A`, isForecast: false });
            lines.push({ word, dim: 'Dominance', color: dColor, dataKey: `${word}_D`, name: `${word} D`, isForecast: false });

            x.forEach((time, index) => {
                let point = workingChartDataMap.get(time) || { time };
                const vVal = typeof v[index] === 'number' && !isNaN(v[index] as number) ? v[index] as number : null;
                const aVal = typeof a[index] === 'number' && !isNaN(a[index] as number) ? a[index] as number : null;
                const dVal = typeof d[index] === 'number' && !isNaN(d[index] as number) ? d[index] as number : null;

                point[`${word}_V`] = vVal;
                point[`${word}_A`] = aVal;
                point[`${word}_D`] = dVal;

                if (vVal !== null) allValues.push(vVal);
                if (aVal !== null) allValues.push(aVal);
                if (dVal !== null) allValues.push(dVal);

                if (showProp && propData && word === singleWord) {
                    point['proportion'] = propData[index];
                }
                workingChartDataMap.set(time, point);
            });
        });

        if (predictedVadSeries && singleWord && commonTimeRef) {
            const word = singleWord;
            const lastActualTimeInData = Math.max(...commonTimeRef);

            const actualDataForWord = allWordsData?.[word]?.temporal_vad;
            let bridgePointCreated = false;

            predictedVadSeries.forEach((pred) => {
                let point = workingChartDataMap.get(pred.time) || { time: pred.time };

                if (!bridgePointCreated && pred.time > lastActualTimeInData && actualDataForWord) {
                    const lastActualDataIndex = actualDataForWord.x.indexOf(lastActualTimeInData);
                    if (lastActualDataIndex !== -1) {
                        let bridgeDataPoint = workingChartDataMap.get(lastActualTimeInData) || {time: lastActualTimeInData};
                        bridgeDataPoint[`${word}_V_forecast`] = actualDataForWord.v[lastActualDataIndex];
                        bridgeDataPoint[`${word}_A_forecast`] = actualDataForWord.a[lastActualDataIndex];
                        bridgeDataPoint[`${word}_D_forecast`] = actualDataForWord.d[lastActualDataIndex];
                        workingChartDataMap.set(lastActualTimeInData, bridgeDataPoint);
                    }
                    bridgePointCreated = true;
                }

                point[`${word}_V_forecast`] = pred.v;
                point[`${word}_A_forecast`] = pred.a;
                point[`${word}_D_forecast`] = pred.d;
                workingChartDataMap.set(pred.time, point);

                if (pred.v !== null) allValues.push(pred.v);
                if (pred.a !== null) allValues.push(pred.a);
                if (pred.d !== null) allValues.push(pred.d);
                if (!commonTimeRef.includes(pred.time)) {
                    commonTimeRef.push(pred.time);
                }
            });

            commonTimeRef.sort((a,b) => a-b);

            const colorBaseIndex = selectedWords.indexOf(word) % lineColors.length;
            lines.push({
                word, dim: 'Valence', color: lineColors[(colorBaseIndex * 3) % lineColors.length],
                dataKey: `${word}_V_forecast`, name: `${word} V (Forecast)`, isForecast: true
            });
            lines.push({
                word, dim: 'Arousal', color: lineColors[(colorBaseIndex * 3 + 1) % lineColors.length],
                dataKey: `${word}_A_forecast`, name: `${word} A (Forecast)`, isForecast: true
            });
            lines.push({
                word, dim: 'Dominance', color: lineColors[(colorBaseIndex * 3 + 2) % lineColors.length],
                dataKey: `${word}_D_forecast`, name: `${word} D (Forecast)`, isForecast: true
            });
        }

        const finalChartData = Array.from(workingChartDataMap.values()).sort((p1, p2) => p1.time - p2.time);

        let finalYDomain: [number | string, number | string] = [0, 1];
        if (allValues.length > 0) {
            const minValue = Math.min(...allValues);
            const maxValue = Math.max(...allValues);
            const range = maxValue - minValue;
            const padding = range < 0.01 ? 0.05 : range * 0.05;
            let calculatedMin = minValue - padding;
            let calculatedMax = maxValue + padding;

            if (calculatedMax - calculatedMin < 0.1) {
                const mid = (calculatedMin + calculatedMax) / 2;
                calculatedMin = mid - 0.05;
                calculatedMax = mid + 0.05;
            }

            finalYDomain = [
                Math.min(calculatedMin, 0),
                Math.max(calculatedMax, 1)
            ];
            if (finalYDomain[0] >= finalYDomain[1]) {
                const center = (finalYDomain[0] + finalYDomain[1]) / 2 || 0.5;
                finalYDomain = [center - 0.1, center + 0.1];
            }
            finalYDomain = [Math.max(finalYDomain[0], -1), Math.min(finalYDomain[1], 1.5)];
            if (finalYDomain[0] >= finalYDomain[1]) finalYDomain = [0,1];


        }

        return {
            chartData: finalChartData,
            yDomain: finalYDomain,
            linesToPlot: lines,
            hasValidData: validDataFound,
            timeDataRef: commonTimeRef || ['auto'],
            showProportionLine: showProp
        };

    }, [selectedWords, allWordsData, selectedSenseData, predictedVadSeries]);


    if (selectedWords.length === 0) {
        return (
            <div className="plot-placeholder info">
                <h4>Select Word(s)</h4>
                <p>Select one or more words from the controls to view the VAD plot.</p>
            </div>
        );
    }

    if (!hasValidData) {
        return (
            <div className="plot-placeholder error">
                <Icon path={mdiAlertCircleOutline} size={1.6} className="placeholder-icon" />
                <h4>Data Error</h4>
                <p>No valid VAD data found for the selected word(s).</p>
                <p className="subtle">Ensure time data aligns across selected words.</p>
            </div>
        );
    }

    const MultiVADTooltip: React.FC<CustomTooltipProps> = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="vad-custom-tooltip multi-line">
                    <p className="label">{`Year: ${label}`}</p>
                    {payload.map((entry: PayloadEntry, index: number) => {
                        if (entry.dataKey === 'proportion') {
                            const value = entry.value;
                            return (
                                <p key={`${index}-prop`} className="desc proportion" style={{ color: 'var(--bs-secondary-color, #6c757d)' }}>
                                    {`Sense Prop.: ${value?.toFixed(3) ?? 'N/A'}`}
                                </p>
                            );
                        } else {
                            const isForecast = entry.dataKey?.endsWith('_forecast');
                            const baseKey = isForecast ? entry.dataKey.slice(0, -9) : entry.dataKey;
                            const keyParts = baseKey.split('_');
                            if (keyParts.length < 2) return null;

                            const word = keyParts[0];
                            const dimLetter = keyParts[1];
                            const dimension = dimLetter === 'V' ? 'Valence' : dimLetter === 'A' ? 'Arousal' : 'Dominance';
                            const value = entry.value;
                            const description = getVADDescription(dimension as VADDimension, value);
                            const nameSuffix = isForecast ? ' (Forecast)' : '';

                            return (
                                <p key={index} className="desc" style={{ color: entry.color, fontStyle: isForecast ? 'italic' : 'normal' }}>
                                    {`${word} ${dimension}${nameSuffix}: ${value?.toFixed(3) ?? 'N/A'}`}
                                    {description && <span className="qualitative">({description})</span>}
                                </p>
                            );
                        }
                    })}
                </div>
            );
        }
        return null;
    };


    return (
        <div className="plot2d-vad-wrapper">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 35, left: 10, bottom: 35 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--grid-color, #e0e0e0)" />
                    <XAxis
                        dataKey="time"
                        type="number"
                        domain={['dataMin', 'dataMax']}
                        allowDuplicatedCategory={false}
                        label={{ value: "Year", position: "insideBottom", dy: 15 }}
                        height={50}
                        scale="time"
                        interval="preserveStartEnd"
                        tickLine={false}
                        ticks={Array.isArray(timeDataRef) && timeDataRef.length > 1 && typeof timeDataRef[0] === 'number' ? timeDataRef as number[] : undefined}
                    />
                    <YAxis
                        yAxisId="left"
                        domain={yDomain}
                        label={{ value: 'VAD Value', angle: -90, position: 'insideLeft', dx: -5 }}
                        width={65}
                        tickFormatter={(tick) => typeof tick === 'number' ? tick.toFixed(2) : String(tick)}
                        allowDataOverflow={true}
                        tickLine={false}
                    />
                    {showProportionLine && (
                        <YAxis
                            yAxisId="right"
                            orientation="right"
                            domain={[0, 1]}
                            label={{ value: 'Sense Prop.', angle: 90, position: 'insideRight', dx: 5 }}
                            width={65}
                            tickFormatter={(tick) => typeof tick === 'number' ? tick.toFixed(2) : String(tick)}
                            tickLine={false}
                        />
                    )}
                    <Tooltip
                        offset={15}
                        cursor={{ stroke: 'var(--text-muted-color)', strokeDasharray: '3 3' }}
                        content={<MultiVADTooltip />}
                    />
                    <Legend verticalAlign="top" height={36} />
                    {linesToPlot.map(lineInfo => (
                        <Line
                            yAxisId="left"
                            key={lineInfo.dataKey}
                            type="monotone"
                            dataKey={lineInfo.dataKey}
                            name={lineInfo.name}
                            stroke={lineInfo.color}
                            strokeWidth={lineInfo.isForecast ? forecastLineStyle.strokeWidth : 1.5}
                            strokeDasharray={lineInfo.isForecast ? forecastLineStyle.strokeDasharray : undefined}
                            activeDot={{ r: 4, strokeWidth: 0, fill: lineInfo.color }}
                            dot={lineInfo.isForecast ? { r: 3, fill: lineInfo.color, strokeWidth:0 } : false}
                            connectNulls={false}
                        />
                    ))}
                    {showProportionLine && (
                        <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="proportion"
                            name="Sense Prop."
                            stroke="var(--bs-secondary-color, #6c757d)"
                            strokeWidth={1}
                            strokeDasharray="5 5"
                            activeDot={{ r: 4, strokeWidth: 0, fill: 'var(--bs-secondary-color, #6c757d)' }}
                            dot={false}
                            connectNulls={false}
                        />
                    )}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}

export default Plot2DVAD;