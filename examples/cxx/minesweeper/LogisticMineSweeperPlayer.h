// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __LogisticMineSweeperPlayer__h__
#define __LogisticMineSweeperPlayer__h__

#include "MineSweeperPlayerBase.h"
#include <utility>
#include <vector>
#include <PUJ_ML/Model/Regression/Logistic.h>

/**
 */
class LogisticMineSweeperPlayer
  : public MineSweeperPlayerBase
{
public:
  using TModel = PUJ_ML::Model::Regression::Logistic< double >;

public:
  LogisticMineSweeperPlayer( );
  virtual ~LogisticMineSweeperPlayer( ) = default;

  virtual void configure(
    unsigned int w, unsigned int h, unsigned long n
    ) override;
  virtual void play( unsigned int& i, unsigned int& j ) override;
  virtual void report( const unsigned char& c ) override;

private:
  unsigned long long _idx(
    const unsigned int& i, const unsigned int& j
    ) const;

protected:
  TModel m_Model;
  TModel::TMatrix m_Inputs;
  std::vector< std::pair< unsigned int, unsigned int > > m_Options;
  unsigned long long m_Choice;
};

#endif // __LogisticMineSweeperPlayer__h__

// eof - $RCSfile$

