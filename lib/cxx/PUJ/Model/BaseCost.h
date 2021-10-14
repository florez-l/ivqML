// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__BaseCost__h__
#define __PUJ__BaseCost__h__

#include <cmath>
#include <vector>

namespace PUJ
{
  namespace Model
  {
    /**
     */
    template< class _TModel >
    class BaseCost
    {
    public:
      using TModel  = _TModel;
      using TMatrix = typename _TModel::TMatrix;
      using TCol    = typename _TModel::TCol;
      using TRow    = typename _TModel::TRow;
      using TScalar = typename _TModel::TScalar;

    public:
      BaseCost(
        TModel* model, const TMatrix& X, const TMatrix& Y,
        unsigned int batch_size = 0
        )
        {
          this->m_Model = model;
          if( batch_size > 0 )
          {
            unsigned int n =
              ( unsigned int )(
                std::ceil( double( X.rows( ) ) / double( batch_size ) )
                );
            for( unsigned int i = 0; i < n; ++i )
            {
              unsigned int bs = batch_size;
              if( ( i + 1 ) * batch_size > X.rows( ) )
                bs = X.rows( ) - ( i * batch_size );
              this->m_X.push_back( X.block( i * batch_size, 0, bs, X.cols( ) ) );
              this->m_Y.push_back( Y.block( i * batch_size, 0, bs, Y.cols( ) ) );
            } // end for
          }
          else
          {
            this->m_X.push_back( X );
            this->m_Y.push_back( Y );
          } // end if
        }

      virtual ~BaseCost( ) = default;

      unsigned int GetNumberOfBatches( ) const
        {
          return( this->m_X.size( ) );
        }

      virtual const TRow& GetParameters( ) const
        {
          return( this->m_Model->GetParameters( ) );
        }
      virtual TScalar operator()(
        unsigned int i, TRow* g = nullptr
        ) const = 0;
      virtual void operator-=( const TRow& g ) = 0;

    protected:
      TModel* m_Model;
      std::vector< TMatrix > m_X;
      std::vector< TMatrix > m_Y;
    };
  } // end namespace
} // end namespace

#endif // __PUJ__BaseCost__h__

// eof - $RCSfile$
